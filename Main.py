import io
import os
import sys
sys.path.append("./unet")# add unet directory to PATH

from skimage import io as im  # io module uses io
import pandas as pd
import numpy as np
import imageio

from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
import cv2
from skimage.morphology import dilation
from skimage import exposure
import argparse

# load modules for prediction, segmentation, and tracking
from track import track, computeLineage
from segment import segment
import neural_network as nn




def save_image(image_stack, fname):
    path = os.path.join(output_path, fname + ".tif")
    im.imsave(path, image_stack, check_contrast=False)


def visualize_mask(masks, raw_images, time=15, auto_close=True, save=True, return_=False):
    '''
    test function used for visualizing masks. Takes image_matrix of type nd-array
    and displays it for 3 seconds. Returns none. Auto_close arg allows user to specify whether
    or not they want to close the pyplot popup menu manually.
    Save: determines whether or not image is saved
    '''

    # ------helper functions------
    def DefineColormap(Ncolors):
        """Define a new colormap by assigning 10 values of the jet colormap
         such that there are only colors for the values 0-10 and the values >10
         will be treated with a modulo operation (updatedata function)
        """
        jet = cm.get_cmap('jet', Ncolors)
        colors = []
        for i in range(0, Ncolors):
            if i == 0:
                # set background transparency to 0
                temp = list(jet(i))
                temp[3] = 0.0
                colors.append(tuple(temp))

            else:
                colors.append(jet(i))

        colormap = ListedColormap(colors)
        return colormap

    def get_img_from_fig(fig):
        '''
        input: matplotlib pyplot figure object
        output: a 3D numpy array object corresponding to the image shown by that figure
        '''
        buf = io.BytesIO()
        fig.savefig(buf, format="tif", bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return (np.array(img)).astype("uint8")

    def showCellNums(mask):
        "annotates the current plt figure with cell numbers"
        #cell_vals = np.unique(mask[mask!=0]).astype(int)
        cell_vals = np.unique(mask).astype(int)
        cell_vals = np.delete(cell_vals, np.where(cell_vals == 0))

        for val in cell_vals:
            # print("cell val: " + str(val)) #test
            x, y = getCenter(mask, val)
            plt.annotate(str(val), xy=(x, y), ha='center', va='center')

    def getCenter(mask, cell_val):
        '''
        takes random points within a cell to estimate its center. Used to find the x,y coordinates where the cell number
        annotation text should be displated
        '''
        y, x = (mask == cell_val).nonzero()
        sample = np.random.choice(len(x), size=20, replace=True)
        meanx = np.mean(x[sample])
        meany = np.mean(y[sample])

        return int(round(meanx)), int(round(meany))
    # ------end helper functions--------

    newcmp = DefineColormap(21)  # initalize colormap
    images_3D = []  # initialize an empty nump array to hold each color mapped, annottated array
    if masks.shape[0] > 1:  # multiple masks
        for t in range(0, masks.shape[0]):
            picture = raw_images[t]
            mask = masks[t]

            # create figure
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(picture, interpolation='None', origin='upper', cmap='gray_r')
            plt.imshow((mask % 10+1)*(mask != 0), origin='upper',
                       interpolation='None', alpha=0.2, cmap=newcmp)
            #plt.imshow(image, interpolation='none', cmap=my_cm, vmin=0.0000001)
            showCellNums(mask)  # annotate
            plt.show(block=False)
            plt.pause(0.01)
            plt.close()

            image_3D = get_img_from_fig(fig)  # get numpy array
            images_3D.append(image_3D)

    else:
        fig = plt.figure()
        plt.imshow(raw_images[0], interpolation='None', origin='upper', cmap='gray_r')
        plt.imshow((masks[0] % 10+1)*(masks[0] != 0), origin='upper',
                   interpolation='None', alpha=0.2, cmap=newcmp)
        showCellNums(masks[0])

        if auto_close:
            plt.show(block=False)
            plt.pause(time)
            plt.close()
        else:
            plt.show()

        images_3D.append(get_img_from_fig(fig))

    # convert to numpy array to get correct dimensions and then save the image
    if save:
        try:
            images_3D = np.array(images_3D)
            save_image(images_3D, filename + "_3D", check_contrast=False)  # filename is a global variable
        except: # in some cases, dimensions do not line up, thus all dimnensions must be set to min dimension of array
            print("WARNING: Differing Image Dimensions Detected Within Figure Representation of Timelapse")
            print("Attempting to Resolve...")
            ax_0 = min(array.shape[0] for array in image_3D)
            ax_1 = min(array.shape[1] for array in images_3D)
            images_3D = np.array([array[:ax_0, :ax_1] if array.shape != (ax_0, ax_1) else array for array in images_3D])
            save_image(images_3D, filename + "_3D")
            print("Resolved - Image has been saved")

    if return_:
        return images_3D


def get_PredThreshSeg(image_matrix, threshold=None, is_pc=False, min_distance=5, topology=None):
    '''
    This function inputs 2D matrix into CNN, thresholds the probability map, and then segments it.
    '''
    # preprocess with "Contrast Limited Adaptive Histogram Equalization" per YeaZ documentation
    image_preprocess = exposure.equalize_adapthist(image_matrix)
    image_preprocess = image_preprocess*1.0

    # run CNN prediction
    prob_map = nn.prediction(image_preprocess, is_pc)

    # threshold image
    thresholded_image = nn.threshold(prob_map, threshold)

    # segment and watershed image
    mask = segment(thresholded_image, prob_map, min_distance, topology)

    return removeCells(mask, cell_size_threshold)  # remove cells below min area threshold


def removeCells(mask, min_area):  # experimental functionality
    '''
    This function takes the mask and removes the cells that have an area less than min area
    '''
    cell_vals = np.unique(mask[mask > 0])
    cell_vals_to_remove = np.array([val for val in cell_vals if (mask == val).sum() < min_area])

    # cell values must be labeled sequentially, no gaps
    for val in cell_vals_to_remove:
        mask[mask == val] = 0
        mask[mask > val] = mask[mask > val] - 1
        cell_vals_to_remove[cell_vals_to_remove >
                            val] = cell_vals_to_remove[cell_vals_to_remove > val] - 1
    return mask



def extractFlou(mask_list, t_start, t_stop, flou_image_path=None):
    if flou_image_path != None:  # an image path was passed into the function, thus flourescence image will be loaded
        # output
        print("------loading flourescence image for flourescence extraction------")

        flou_image_stack = np.array(im.MultiImage(flou_image_path))  # load flourescence image as numpy array

        # format 2D array from [n,m] into [1,m,n] for easier processing and saving
        if flou_image_stack.ndim == 2:
            flou_image_stack = np.array([flou_image_stack])
    else:
        flou_image_stack = image_stack  # global variable defined by user input

    # get cell stats for each cell, t_value pair
    cell_list = []  # a list of dictionaries containing cell stats for cell/time
    for t in range(t_start, t_stop):  # iterate through time values
        lst_index = t-t_start  # ie: an idex of 0 corresponds to t = t_start

        # get mask for specific value of t
        mask_t = mask_list[lst_index]
        image_t = flou_image_stack[lst_index]

        # for some value of t, get cell stats for each unique cell_val
        for cell_val in np.unique(mask_t):  # iterate through each cell value
            # bg is not cell
            if cell_val == 0:  # background = 0
                continue

            # Calculate stats for some value of cell_val
            stats = {'Cell': cell_val,
                     'Time': t}

            # ** denotes merge dictionaries
            stats = {**stats,
                     **cell_statistics(image_t, mask_t, cell_val)}
            stats['Disappeared in video'] = not (cell_val in np.unique(mask_t))
            cell_list.append(stats)

    # Use Pandas to write csv
    filename_csv = filename + "_flou.csv"
    print("-----writing cell data to " + filename_csv + "------")  # test
    df = pd.DataFrame(cell_list)
    df = df.sort_values(['Cell', 'Time'])
    path = os.path.join(output_path, filename_csv)
    df.to_csv(path, index=False)


def cell_statistics(image, mask, cell_val):
    """Calculate statistics about cells. Passing None to image will
    create dictionary to zeros, which allows to extract dictionary keys"""
    if image is not None:
        cell_vals = image[mask == cell_val]
        area = (mask == cell_val).sum()
        tot_intensity = cell_vals.sum()
        mean = tot_intensity/area if area > 0 else 0
        var = np.var(cell_vals)

        # Center of mass
        y, x = (mask == cell_val).nonzero()
        com_x = np.mean(x)
        com_y = np.mean(y)

        # PCA to determine morphology
        pca = PCA().fit(np.array([y, x]).T)
        pc1_x, pc1_y = pca.components_[0, :]
        angle = np.arctan(pc1_y / pc1_x) / np.pi * 360
        v1, v2 = pca.explained_variance_

        len_maj = 4*np.sqrt(v1)
        len_min = 4*np.sqrt(v2)

    else:
        mean = 0
        var = np.nan
        tot_intensity = 0
        com_x = np.nan
        com_y = np.nan
        angle = np.nan
        len_maj = np.nan
        len_min = np.nan

    return {'Area': area,
            'Mean': mean,
            'Variance': var,
            'Total Intensity': tot_intensity,
            'Center of Mass X': com_x,
            'Center of Mass Y': com_y,
            'Angle of Major Axis': angle,
            'Length Major Axis': len_maj,
            'Length Minor Axis': len_min}


if __name__ == '__main__':


    # set up argparse
    parser = argparse.ArgumentParser(
        description="WELCOME TO YEASTPED \n Use the command line to input parameters for cell tracking and segmentation. Refer to userguide for further documentation", fromfile_prefix_chars="@")

    # segmentation parameters
    parser.add_argument("-image_path", type=str, required=True,
                        help = "path to raw images (useful for visualization even if no segmentation is desired)")
    parser.add_argument("-segmented_image_path", type=str,
                        help="specifying an image path for -segmentated_images_path will bypass segmentation. Only tracking and lineage tracking will be performed")
    parser.add_argument("-output_dir", type=str, required=True,
                        help="directory where you want output to be located. Will be created if inputted path does not exist")
    parser.add_argument("-output_filename", type=str, default="output",
                        help="name prefix given to all files created. Default is 'output'")
    parser.add_argument("-start_frame", type=int, default=0, help="frame to begin analysis")
    parser.add_argument("-end_frame", type=int, help="frame to end analysis")
    parser.add_argument("-threshold", type=float, default=0.50, help="segmentation probability threshold (YeaZ)")
    parser.add_argument("-seed_distance", type=int, default=5, help="min watershed seed distance (YeaZ)")
    parser.add_argument("-get_flou", action="store_true",
                        help="extract flourescence and cell statistics")
    parser.add_argument("-flou_image_path", type=str, help="path to the flourescence image.")

    # cost function parameters
    parser.add_argument("-weight_overlap", type=int, default=95, help="Cost function weight for cell area overlap")
    parser.add_argument("-weight_centroids", type=int, default=50, help="Cost function weight for cell-to-cell centroid distance")
    parser.add_argument("-weight_size", type=int, default=10, "Cost function weight for cell size similarity from frame to frame")
    parser.add_argument("-max_centroids_distance", type=float, default=75.0, "Max distance between centroids of identical cells")

    # mitosis parameters
    parser.add_argument("-division_overlap_threshold", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument("-daughter_size_similarity", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("-daughter_aspect_ratio_similarity", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("-circularity_threshold", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("-number_of_frames_check_circularity", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("-disable_cell_mitosis", action="store_false", help=argparse.SUPPRESS)

    # confidence index parameters
    parser.add_argument("-cell_life_threshold", type=int, default=20,
                        help="Cells with lifespan's less than this value will be removed. Cells born near the end of time-lapse are exempt from removal")
    parser.add_argument("-cell_apoptosis_delta_centroid_thres", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("-cell_density_ci_flag", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("-border_cell_ci_flag", action="store_true", help=argparse.SUPPRESS)

    # fusion parameters
    parser.add_argument("-cell_size_threshold", type=int, default=100,
                        help="Cells with areas less than this threshold are considered artefact and discarded")
    parser.add_argument("-enable_cell_fusion_flag", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-fusion_overlap_threshold", type=int, default=50,
                        help="overlap threshold over which cells two cells are considered to be incorrectly fused")

    args = parser.parse_args()

    # initalize segmentation variables
    image_path = args.image_path  # global variable used in ExtractFlou function
    output_path = args.output_dir # global variable used in ExtractFlou function and save_image function
    if output_path is None:
        output_path = image_path
    elif not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = args.output_filename #GLOBAL - used by extract flou and visualize mask

    seg_images = args.segmented_image_path
    if not os.path.exists(seg_images):
        print("------ERROR:" + str(seg_images) + " does not not exit-------")
        exit()

    seg_bool = seg_images==None

    t_start = args.start_frame
    t_stop = args.end_frame
    th = args.threshold
    min_d = args.seed_distance  # minimum distance between watershed seeds
    extract_flou_bool = args.get_flou  # boolean
    flourescence_im_path = args.flou_image_path
    if not os.path.exists(flourescence_im_path):
        print("------ERROR:" + str(flourescence_im_path) + " does not not exit-------")
        exit()


    ###LINEAGE MAPPER VARIABLES###
    # File compatibilities
    segmented_images_path = os.path.join(output_path, "segmented")
    if not os.path.exists(segmented_images_path):
        os.makedirs(segmented_images_path)
    segmented_images_common_name = "seg_"

    # tracked images will be located in the tracked folder in the output path
    tracked_images_path = os.path.join(output_path, "tracked")
    if not os.path.exists(tracked_images_path):
        os.makedirs(tracked_images_path)
    tracked_images_common_name = "tracked_"

    # cost_function parameters
    weight_overlap = args.weight_overlap
    weight_centroids = args.weight_centroids
    weight_size = args.weight_size
    max_centroids_distance = args.max_centroids_distance
    frames_to_track_nb = 0  # all frames will be tracked - signified by a 0 in LM

    # mitotic paramters
    division_overlap_threshold = args.division_overlap_threshold
    daughter_size_similarity = args.daughter_size_similarity
    daughter_aspect_ratio_similarity = args.daughter_aspect_ratio_similarity
    number_of_frames_check_circularity = args.number_of_frames_check_circularity
    circularity_threshold = args.circularity_threshold
    enable_cell_mitosis_flag = args.disable_cell_mitosis

    # confidence interval parameters
    cell_life_threshold = args.cell_life_threshold
    cell_apoptosis_delta_centroid_thres = args.cell_apoptosis_delta_centroid_thres
    cell_density_ci_flag = args.cell_density_ci_flag
    border_cell_ci_flag = args.border_cell_ci_flag

    # fusion parameters
    enable_cell_fusion_flag = args.enable_cell_fusion_flag
    fusion_overlap_threshold = args.fusion_overlap_threshold
    cell_size_threshold = args.cell_size_threshold

    #load the image
    image_stack = np.array(im.MultiImage(image_path))

    #configure t_stop
    if t_stop == None:
        t_stop = len(image_stack)

    # non user parameters
    nb_frames = (t_stop - t_start)
    max_cell_num = 12000 # lineage mapper parameter

    # format array from [n,m] into [1,m,n] for easier processing and saving
    if image_stack.ndim == 2:
        image_stack = np.array([image_stack])
        cell_track_bool = False  # single image - no tracking
        t_stop = t_stop + 1  # allow iteration over single mask
    else:
        cell_track_bool = True  # multi_stack image


    # output
    print("------input image shape------")
    print(image_stack.shape)


    # get image masks for each 2D array in image_stack:
    if seg_bool:
        mask_list = []
        call_number = 0
        for image in image_stack[t_start:t_stop]:  # iterate through each image
            print("------generating mask for image: " +
                  str(call_number) + "------")
            #  generate a mask for each image and append it to mask_list
            mask_list.append(get_PredThreshSeg(image, threshold=th, min_distance=min_d))
            call_number += 1
    else:
        mask_list = im.MultiImage(seg_images)
        mask_list = np.array([removeCells(mask, cell_size_threshold) for mask in mask_list])
        print("------segmented image input detected------")
        print("------segmented image shape: " + str(mask_list.shape))


    # addditional functionalities specified by user:
    if cell_track_bool:
        mask_list = track(mask_list, segmented_images_path, segmented_images_common_name, tracked_images_path,
                          tracked_images_common_name, max_centroids_distance, weight_size, weight_centroids,
                          weight_overlap, fusion_overlap_threshold, division_overlap_threshold,
                          cell_life_threshold, max_cell_num, nb_frames, daughter_size_similarity, daughter_aspect_ratio_similarity,
                          cell_size_threshold, enable_cell_fusion_flag, frames_to_track_nb, cell_density_ci_flag, border_cell_ci_flag,
                          number_of_frames_check_circularity, circularity_threshold, cell_apoptosis_delta_centroid_thres, enable_cell_mitosis_flag)
        computeLineage(mask_list, output_path)
    if extract_flou_bool:
        extractFlou(mask_list, t_start, t_stop, flou_image_path=flourescence_im_path)


    save_image(np.array(mask_list), fname=(filename + "_masks"))

    print("-----Showing resulting image stack")
    visualize_mask(np.array(mask_list), image_stack, save=True)
