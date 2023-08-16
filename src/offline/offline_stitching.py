import numpy
import torch
import cv2
from os import listdir
from os.path import join, basename, splitext

from typing import Callable
from models.superpoint import SuperPointMinimal
from preprocessing.averaging import calculate_profile, apply_profile_to_image
from configuration import configuration
from matplotlib import pyplot as plt

torch.set_grad_enabled(False)


class OfflineStitcher(object):
    def __init__(self):
        self.superpoint = SuperPointMinimal()
        self.superglue = None
        self.input_directory = None
        self.output_directory = None
        self.working_title = None
        self.images = []
        self.interest_points = []
        self.matching_scores = []

    def set_input_directory(self, path: str):
        self.input_directory = path
        
    def set_output_directory(self, path: str):
        self.output_directory = path

    def set_working_title(self, working_title: str):
        self.working_title = working_title

    def load_images(self, predicate: Callable[[str], bool] = lambda filename: splitext(filename)[1][1:] in ["tif"]):
        self.images = [cv2.imread(join(self.input_directory, filename),cv2.IMREAD_GRAYSCALE) for filename in listdir(self.input_directory) if
                       predicate(filename)]

    def preprocess_images(self, smooth_steps: int = 0):
        average_profile = calculate_profile(self.images, smooth_steps)
        self.images = [apply_profile_to_image(image, average_profile) for image in self.images]

    def compute_raw_interest_points_and_descriptors(self):
        for image in self.images:
            self.superpoint.encode(self.frame_to_tensor(image, configuration.device))
            self.superpoint.compute_scores()
            self.superpoint.extract_keypoints()
            #self.superpoint.extract_keypoint_scores()
            #self.superpoint.remove_keypoints_at_image_border()
            #if self.superpoint.max_keypoints >= 0:
            #    self.superpoint.filter_top_k_keypoints()
            self.superpoint.flip_keypoints()
            self.superpoint.compute_descriptors()
            self.superpoint.extract_descriptors()
            self.interest_points.append(self.superpoint.keypoints.copy())
            self.superpoint.reset_state()

    def save_raw_interest_points_and_descriptors(self, path: str):
        pass

    def load_raw_interest_points_and_descriptors(self, path: str):
        pass

    def filter_interest_points_and_descriptors(self):
        pass


    @staticmethod
    def frame_to_tensor(frame: numpy.ndarray, device: str) -> torch.Tensor:
        """
        Reshapes the input image with dimension (height,width) to (batch_size,channels,height,width) for subsequent use
        as input to Superpoint, where batch_size = channels = 1
        """
        return torch.from_numpy(frame / 255.).float()[None][None].to(device)




