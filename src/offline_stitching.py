import os.path
from dataclasses import dataclass, field
import cv2
import numpy
import torch
from os import listdir
from os.path import join, splitext, basename
from models.matching import Matching, MatchingInput, MatchingOutput
from models.superglue import SuperGlueConfig, SuperGlue, SuperGlueInput, SuperGlueOutput
from models.superpoint import SuperPointConfig, SuperPoint, SuperPointInput, SuperPointOutput
from util.metrics import capture_timing_info

torch.set_grad_enabled(False)


@dataclass
class Config:
    input_directory: str
    output_directory: str
    nms_radius: int = 4
    keypoint_threshold: float = 0.005
    max_keypoints: int = 1024
    superglue_weights: str = "indoor"
    sinkhorn_iterations: int = 100
    match_threshold: float = 0.80
    force_cpu: bool = False
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    supported_file_extensions: list[str] = field(default_factory=lambda: ["jpg", "tif"])


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __abs__(self):
        return Point(abs(self.x), abs(self.y))


def frame_to_tensor(frame: numpy.ndarray, device: str) -> torch.Tensor:
    """
    Reshapes the input image with dimension (height,width) to (batch_size,channels,height,width) for subsequent use
    as input to Superpoint, where batch_size = channels = 1
    """
    return torch.from_numpy(frame / 255.).float()[None][None].to(device)


@capture_timing_info()
def generate_keypoints(matcher, image_a: numpy.ndarray, image_b: numpy.ndarray, device: str) -> \
        tuple[numpy.ndarray, numpy.ndarray]:

    tensor_a = frame_to_tensor(image_a, device)
    tensor_b = frame_to_tensor(image_b, device)

    prediction: MatchingOutput = matcher(MatchingInput(image0=tensor_a, image1=tensor_b))
    kpts0, kpts1 = prediction.keypoints0[0].cpu().numpy(), prediction.keypoints1[0].cpu().numpy()
    matches, confidence = prediction.matches0[0].cpu().numpy(), prediction.matching_scores0[0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    return mkpts0, mkpts1


class OfflineStitcher(object):
    def __init__(self, superpoint_config, superglue_config):
        self.superpoint = SuperPoint(superpoint_config)
        self.superglue = SuperGlue(superglue_config)

    def extract_interest_point_data(self, image: numpy.ndarray):
        image_tensor = frame_to_tensor(image)
        result: SuperPointOutput = self.superpoint(SuperPointInput(image_tensor))
        # Serialize for later use
        # TODO(Leah): implement serialization + loading from file

    def match_interest_point_data(self, interest_points_a, interest_points_b):
        pass

    def create_stitch(self):
        pass
