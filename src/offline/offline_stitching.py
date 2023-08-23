import numpy
import torch
import cv2

from models.superpoint import SuperPointMinimal
from preprocessing.averaging import calculate_profile, apply_profile_to_image
from configuration import configuration
from matplotlib import pyplot as plt
import pickle

from util.helpers import get_file_paths_with_extensions

torch.set_grad_enabled(False)


class OfflineStitcher(object):
    def __init__(self):
        self.superpoint = SuperPointMinimal()
        self.superglue = None
        self.input_directory = None
        self.output_directory = None
        self.working_title = None
        self.images = []
        self.interest_point_data = []


    def set_input_directory(self, path: str):
        self.input_directory = path
        
    def set_output_directory(self, path: str):
        self.output_directory = path

    def set_working_title(self, working_title: str):
        self.working_title = working_title

    def load_images(self):
        self.images = [cv2.imread(image_path,cv2.IMREAD_GRAYSCALE) for image_path in get_file_paths_with_extensions(self.input_directory, configuration.supported_file_extensions)]

    def preprocess_images(self, smooth_steps: int = 0):
        average_profile = calculate_profile(self.images, smooth_steps)
        self.images = [apply_profile_to_image(image, average_profile) for image in self.images]

    def compute_raw_interest_points_and_descriptors(self):
        for image in self.images:
            scores, descriptors = self.superpoint(self.frame_to_tensor(image, configuration.device))
            self.interest_point_data.append((image, scores.cpu().numpy(), descriptors.cpu().numpy()))

    def save_raw_interest_points_and_descriptors(self, path: str):
        with open(path,"w+") as file:
            pickle.dump(self.interest_point_data,file)

    def load_raw_interest_points_and_descriptors(self, path: str):
        with open(path,"w+") as file:
            self.interest_point_data = pickle.load(file)

    def filter_interest_points_and_descriptors(self):
        pass


    @staticmethod
    def frame_to_tensor(frame: numpy.ndarray, device: str) -> torch.Tensor:
        """
        Reshapes the input image with dimension (height,width) to (batch_size,channels,height,width) for subsequent use
        as input to Superpoint, where batch_size = channels = 1
        """
        return torch.from_numpy(frame / 255.).float()[None][None].to(device)




