import numpy
import torch
import cv2
import pickle
from os.path import join, exists
from os import makedirs

from offline.models.superpoint_minimal import SuperPointMinimal
from preprocessing.averaging import calculate_profile, apply_profile_to_image
from configuration import global_configuration
from matplotlib import pyplot as plt
from util.helpers import get_file_paths_with_extensions
from offline.models.superglue import SuperGlue, SuperGlueInput, SuperGlueOutput

torch.set_grad_enabled(False)

class OfflineStitcher(object):
    def __init__(self,load_models=False):
        self.superpoint = None
        self.superglue = None

        self.input_directory = None
        self.output_directory = None
        self.working_title = None
        self.keypoint_score_threshold = None
        self.max_keypoint_count = None
        
        self.images = []
        self.interest_point_data = []
        self.matching_list = []

        if load_models:
            self.load_superpoint()
            self.load_superglue()

    def load_superpoint(self):
        self.superpoint = SuperPointMinimal()

    def load_superglue(self):
        self.superglue = SuperGlue()

    def set_input_directory(self, path: str):
        self.input_directory = path
        
    def set_output_directory(self, path: str):
        self.output_directory = path

    def set_working_title(self, working_title: str):
        self.working_title = working_title

    def load_images(self):
        self.images = [cv2.imread(image_path,cv2.IMREAD_GRAYSCALE) for image_path in get_file_paths_with_extensions(self.input_directory, global_configuration.supported_file_extensions)]

    def preprocess_images(self, smooth_steps: int = 0):
        average_profile = calculate_profile(self.images, smooth_steps)
        self.images = [apply_profile_to_image(image, average_profile) for image in self.images]

    def compute_interest_points_and_descriptors(self, keypoint_score_threshold: float, max_keypoint_count: int):
        self.keypoint_score_threshold = keypoint_score_threshold
        self.max_keypoint_count = max_keypoint_count
        #NOTE(Leah): Add keypoint border exclusion code here 
        for image in self.images:
            scores, descriptors = self.superpoint(self.numpy_image_data_to_tensor(image, global_configuration.device))
            
            scores = scores.squeeze()
            descriptors = descriptors.squeeze()

            coordinates = torch.nonzero(scores > keypoint_score_threshold)
            corresponding_scores = scores[coordinates[:,0],coordinates[:,1]]

            final_scores, indices = torch.topk(corresponding_scores, min(max_keypoint_count,len(corresponding_scores)))
            final_coordinates = coordinates[indices]
        
            final_descriptors = descriptors[:,final_coordinates[:,0],final_coordinates[:,1]].permute((1,0))
        
            self.interest_point_data.append((image, final_scores, final_coordinates, final_descriptors))
        
    def save_interest_point_data(self, filename: str):
        directory = join(self.output_directory,self.working_title)
        if not exists(directory):
            makedirs(directory)
        with open(join(directory, filename),"wb+") as file:
            pickle.dump(self.interest_point_data, file)

    def load_interest_point_data(self, filename: str):
        path = join(self.output_directory,self.working_title,filename)
        with open(path,"rb") as file:
            self.interest_point_data = pickle.load(file)

    def visualize_keypoints(self,index: int):
        image,scores,coordinates,_ = self.interest_point_data[index]
        scores = scores.cpu().numpy()
        coordinates = coordinates.cpu().numpy()
        plt.imshow(image,cmap="Greys_r")
        plt.scatter(coordinates[:,1],coordinates[:,0],c="maroon",s=scores*10)
        plt.show()

    def set_matching_list(self, matching_list: tuple[int,int]):
        self.matching_list = matching_list

    def match(self, index_a: int, index_b: int):
        pass



    @staticmethod
    def numpy_image_data_to_tensor(frame: numpy.ndarray, device: str) -> torch.Tensor:
        """
        Reshapes the input image with dimension (height,width) to (batch_size,channels,height,width) for subsequent use
        as input to Superpoint, where batch_size = channels = 1
        """
        return torch.from_numpy(frame / 255.).float()[None][None].to(device)




