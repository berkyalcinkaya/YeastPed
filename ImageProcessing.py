import os
from skimage import io
import numpy as np
import cv2
from matplotlib import pyplot as plt


def splitStackTif(input, output_path, image_name="im"):
    if not (os.path.exists(output_path)):
        os.makedirs(output_path)
        print("------Path does not exist. Creating new dirs...")

    if type(input) == str:  # filepath was input
        image_stack = io.imread(nparray)
    else:  # image stack was input
        image_stack = input

    counter = 0
    for image in image_stack:
        if image is not None:
            if image_name.endswith("_"):
                _seperator = ""
            else:
                _seperator = "_"

            io.imsave(os.path.join(output_path, image_name +
                      _seperator + str(counter).zfill(3) + ".tif"), image, check_contrast=False)
            counter += 1



def makeStackTif(image_path, com_name, returnList=False, output_path=os.getcwd(), output_name="stack"):
    if not returnList and not (os.path.exists(output_path)):
        os.makedirs(output_path)
        print("------Path does not exist. Creating new dirs...")

    image_list = []
    files = os.listdir(image_path)
    for file in sorted(files):
        if com_name in file:
            image = io.imread(os.path.join(image_path, file))
            image_list.append(image)
    image_list = np.array(image_list, dtype='int16')

    if returnList:
        return image_list
    else:
        io.imsave(os.path.join(output_path, output_name + ".tif"), image_list, check_contrast=False)
        return None
