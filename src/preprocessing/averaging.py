import cv2
from os.path import join, normpath, basename, exists
from os import listdir, makedirs

import numpy as np
from src.util.metrics import capture_timing_info, print_metrics


@capture_timing_info()
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


@capture_timing_info()
def apply_average_normalization(input_directory, output_directory):
    image_paths = [join(normpath(input_directory), filename) for filename in listdir(input_directory)]
    image_tuples = [(basename(image_path), cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255.0) for image_path in image_paths]
    images = np.array(list(map(lambda image_tuple: image_tuple[1], image_tuples)))
    average_image = np.average(images, axis=0)
    normalized_images = list(
        map(lambda image: (image[0], normalize(np.divide(image[1], average_image))), image_tuples))

    if not exists(output_directory):
        makedirs(output_directory)

    for image_name, normalized_image in normalized_images:
        cv2.imwrite(join(output_directory, image_name), (normalized_image*255.0).astype(np.ubyte))
