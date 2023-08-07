import os.path
from dataclasses import dataclass, field
import cv2
import numpy
import torch

from os import listdir
from os.path import join, splitext, basename

from models.superglue import SuperGlueConfig, SuperGlue, SuperGlueInput, SuperGlueOutput
from models.superpoint import SuperPointConfig, SuperPoint, SuperPointInput, SuperPointOutput

from configuration import configuration
import pickle

torch.set_grad_enabled(False)


@dataclass
class Image:
    path: str
    data: numpy.ndarray


@dataclass
class OfflineStitcher(object):
    superpoint: SuperPoint
    superglue: SuperGlue
    images: list[Image] = field(default_factory=lambda: [])
    superpoint_outputs: list[tuple[str, SuperPointOutput]] = field(default_factory=lambda: [])

    def load_images_from_directory(self, directory_path: str):
        filepaths = [os.path.realpath(join(directory_path, filename)) for filename in listdir(directory_path) if
                     splitext(filename)[1][1:] in configuration.supported_file_extensions]
        self.images = [Image(path, cv2.imread(path, cv2.IMREAD_GRAYSCALE)) for path in filepaths]

    def extract_keypoints_from_images(self):
        for image in self.images:
            self.superpoint_outputs.append((image.path, self.superpoint(SuperPointInput(image=OfflineStitcher.frame_to_tensor(image.data, configuration.device)))))

    def run(self, input_directory: str, output_directory: str):
        self.load_images_from_directory(input_directory)
        self.extract_keypoints_from_images()
        with open(join(output_directory, "keypoint_data.pickle"), "wb+") as pickle_out_file:
            pickle.dump(self.superpoint_outputs, pickle_out_file)
        self.filter_keypoints()
        self.generate_pairs()
        self.match()
        self.save_matches()
        self.filter_matches()

        # Assume all preprocessing has been done already and the images are usable
        # Capture all keypoints of every image and save them to disk for further examination
        # Do Filtering on the Keypoints as neccessary
        # Create Pairs of Keypoints and Match them via SuperGlue
        # Here we should implement subregion stitching
        # Capture all matches/scores and write them to disk for further examination (image data + pure data)
        # Filter all The Matches and pick the best ones
        # Filter the matching keypoints
        # generate affine Transforms for each match
        # apply affine transform to each image
        # do the final image stitch
        pass

    @staticmethod
    def frame_to_tensor(frame: numpy.ndarray, device: str) -> torch.Tensor:
        """
        Reshapes the input image with dimension (height,width) to (batch_size,channels,height,width) for subsequent use
        as input to Superpoint, where batch_size = channels = 1
        """
        return torch.from_numpy(frame / 255.).float()[None][None].to(device)
