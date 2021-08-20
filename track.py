import os
import numpy as np
import pandas as pd
import cv2
from ImageProcessing import splitStackTif, makeStackTif
from skimage.morphology import dilation
from computeMajorAxisVectorOverlap import computeMajorAxisVectorOverlap as axOverlap
from skimage import io
import matlab.engine


def track(segmented_masks, segmented_images_path, segmented_images_common_name, tracked_images_path,
          tracked_images_common_name, max_centroids_distance, weight_size, weight_centroids,
          weight_overlap, fusion_overlap_threshold, division_overlap_threshold,
          cell_life_threshold, max_cell_num, nb_frames, daughter_size_similarity, daughter_aspect_ratio_similarity,
          cell_size_threshold, enable_cell_fusion_flag, frames_to_track_nb, cell_density_ci_flag, border_cell_ci_flag,
          number_of_frames_check_circularity, circularity_threshold, cell_apoptosis_delta_centroid_thres, enable_cell_mitosis_flag):
    "params: see Lineage Mapper parameters"
    "output: a 3D array of tracked masks"

    # split the 3D np array into single tifs with a common name - save to segmented images path
    splitStackTif(segmented_masks, output_path=segmented_images_path,
                  image_name=segmented_images_common_name)

    # call start_tracking.m
    eng = matlab.engine.start_matlab()
    eng.addpath("./LM_tracking", nargout=0)
    tracking_completed = eng.start_tracking(segmented_images_path, segmented_images_common_name, tracked_images_path,
                                            tracked_images_common_name, max_centroids_distance, weight_size/100, weight_centroids/100,
                                            weight_overlap/100, fusion_overlap_threshold/100, division_overlap_threshold/100,
                                            cell_life_threshold, max_cell_num, nb_frames, daughter_size_similarity/100,
                                            daughter_aspect_ratio_similarity/100, cell_size_threshold, enable_cell_fusion_flag,
                                            frames_to_track_nb, cell_density_ci_flag, border_cell_ci_flag,
                                            number_of_frames_check_circularity, circularity_threshold/100, cell_apoptosis_delta_centroid_thres, enable_cell_mitosis_flag)
    eng.quit()
    LM_output_stack = makeStackTif(image_path=tracked_images_path, com_name=tracked_images_common_name, returnList=True)

    #remove cells with minimum lifespan
    masks_tracked = removeMinLifeSpan(tracked_images_path, LM_output_stack, cell_life_threshold)

    return masks_tracked


def removeMinLifeSpan(tracked_images_path, tracked_mask_list, cell_life_threshold):
    birth_path = os.path.join(tracked_images_path, "birth.csv")
    birth_df = pd.read_csv(birth_path)

    death_path = os.path.join(tracked_images_path, "death.csv")
    death_df = pd.read_csv(death_path)

    merged_df = pd.merge(birth_df, death_df, on="Cell ID")
    merged_df[["Birth Frame", "Death Frame"]] = merged_df[["Birth Frame", "Death Frame"]] - 1 # convert to zero indexing
    merged_df["Life Span"] = merged_df["Death Frame"] - merged_df["Birth Frame"] + 1

    cells_remove_df = merged_df.loc[(merged_df["Life Span"] <= cell_life_threshold) & (
    merged_df["Death Frame"] != merged_df["Death Frame"].max())]

    cells_remove_df.to_csv(os.path.join(tracked_images_path, "cells_removed.csv"))
    cells_remove_lst = np.array(cells_remove_df["Cell ID"].tolist())

    for cell_val in cells_remove_lst:
        # set all pixels with value cell_val to 0, bump the rest of the values down by one
        tracked_mask_list = np.where(tracked_mask_list == cell_val, 0, tracked_mask_list)
        tracked_mask_list = np.where(tracked_mask_list > cell_val, tracked_mask_list-1, tracked_mask_list)

        # update cells_remove_lst
        cells_remove_lst[cells_remove_lst >
                         cell_val] = cells_remove_lst[cells_remove_lst > cell_val] - 1
    return tracked_mask_list


def computeLineage(array_3D, output_path):
    '''
    input:
    array_3D - stack of masks in np array format
    output_path - path to which lineage information in csv format is written

    output:
    returns none

    descrip: finds each cell's mother and writes to csv
    '''
    print("------computing lineage------")
# -----------helper functions
    def checkLifeSpan(potential_mother_cell_val,idx_y_min,lifespan_requirement=1):
        '''
        for a given potential mother, checks to see if the cell is eligible to be a mother based on
        lifespan
        output: bool
        '''
        potential_mother_birth_frame = np.min((np.nonzero(array_3D == potential_mother_cell_val))[0])
        if potential_mother_birth_frame == 0:
            return True
        else:
            return (idx_y_min - potential_mother_birth_frame >= lifespan_requirement)


    def checkArea(potential_mother_cell_val, daughter_cell_val, idx_y_min):
        return (array_3D[idx_y_min]==potential_mother_cell_val).sum() > (array_3D[idx_y_min]==daughter_cell_val).sum()


    def getMother(cell_val):
        '''
        helper function
        input: cell_val whose lineage is to be determined.
        Note: this function relies on having access to array_3D which is the 3D array of masks
        output: returns the mother, birth frame, and death_frame of cell_val

        descrip: finds the birth frame of cell_val, dilates the cell, and indentifies the mother as the cell that overlaps
        the most with dilated mask of cell_val
        '''
        idx_array = np.nonzero(array_3D == cell_val)  # returns 3 lists corresponding to the 3 dimensions of the input array, each list contains that dimension's indices where the value exists
        idx_y_min = int(np.min(idx_array[0]))  # find the first frame that cell_val appears
        idx_y_max = int(np.max(idx_array[0]))  # death frame


        # cell is present at the beginning of time lapse
        if idx_y_min == 0:
            mother = "Ori"
            major_axis_cells = "N/A"
            return idx_y_min, idx_y_max, mother, major_axis_cells

        # frames to be examined for overlap
        birth_frame = array_3D[idx_y_min]

        #handle cells born at the end of movie
        try:
            prebirth_frames = array_3D[idx_y_min-2:idx_y_min+2, :, :]
        except:
            prebirth_frames = array_3D[idx_y_min-2:idx_y_min+1, :, :]

        mother_overlap_counts = {}  # structure -- (val that overlaps with cell_val):(pixels of overlap)
        for frame in prebirth_frames:
            x = 15  # inital kernel is 15x15
            kernel = np.ones((x, x), dtype=bool)
            cell_dilation = dilation(birth_frame == cell_val, kernel)
            overlapping_vals, count = np.unique(frame[cell_dilation], return_counts=True)
            potential_mothers = [val for val in overlapping_vals[np.logical_and(overlapping_vals != 0, overlapping_vals != cell_val)]
                                                         if checkLifeSpan(potential_mother_cell_val=val, idx_y_min=idx_y_min) and
                                                         checkArea(potential_mother_cell_val = val, daughter_cell_val=cell_val, idx_y_min=idx_y_min)]
            #update mother_overlap_counts
            for idx, value in enumerate(overlapping_vals):
                if value in mother_overlap_counts:  # cell value is alr in the dictionary
                    mother_overlap_counts[value] += count[idx]
                elif value in potential_mothers:  # value is not yet in the dictionary, test it's presence in potential mothers
                    mother_overlap_counts[value] = count[idx]


        ## find the mother
        #if only one mother was detected, no major axis overlap is computed
        if len(mother_overlap_counts) == 1:
            mother = (list(mother_overlap_counts))[0]
            major_axis_cells = "N/A"
        else:
            major_axis_cells = axOverlap(cell_val = cell_val, birth_idx = idx_y_min, masks = array_3D)

            #cells were overlapping the major axis
            if len(major_axis_cells)>0:
                mother_overlap_counts = {k:v for k,v in mother_overlap_counts.items() if k in major_axis_cells}
                mother = keywithmaxval(dict = mother_overlap_counts)
                major_axis_cells = lst_to_str(major_axis_cells)

            #no cells were consistently detected along major axis
            else:
                mother = keywithmaxval(dict = mother_overlap_counts)
                major_axis_cells = "*"

        return idx_y_min, idx_y_max, mother, major_axis_cells

# ---------------end helper function
    cell_lineage_list = []
    for cell_val in np.unique(array_3D[array_3D > 0]):
        birth_frame, death_frame, mother, ellipse_cells = getMother(cell_val)
        cell_life_dict = {"Cell": cell_val, "Birth Frame": birth_frame,
                          "Death Frame": death_frame, "Mother": mother, "Major Axis Overlap": ellipse_cells, "Correct": " "}
        cell_lineage_list.append(cell_life_dict)

    # Use Pandas to write csv
    filename_pedigree = "lineage.csv"
    lineage_output_path = os.path.join(output_path, filename_pedigree)
    print("------writing lineage data to " + lineage_output_path + "------")  # test
    df = pd.DataFrame(cell_lineage_list)
    df = df.sort_values("Cell")
    df.to_csv(lineage_output_path, index=False)

    return None


def get_key(value, dict):
    '''
    given a value, get_key locates the first occurence of a key with the given value
    '''
    for key, val in dict.items():
        if val == value:
            return key


def keywithmaxval(dict):
    """ a) create a list of the dict's keys and values;
     b) return the key with the max value"""
    if len(dict) > 0:
        v=list(dict.values())
        k=list(dict.keys())
        return k[v.index(max(v))]
    else:
        return "Check"


def lst_to_str(lst):
    return (",".join(map(str, lst)))
