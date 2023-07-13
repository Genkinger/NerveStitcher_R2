import os.path
from dataclasses import dataclass, field
import cv2
import numpy
import torch
from os import listdir
from os.path import join, splitext, basename
from models.matching import Matching, MatchingInput, MatchingOutput
from models.superglue import SuperGlueConfig
from models.superpoint import SuperPointConfig
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


def frame_to_tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


@capture_timing_info()
def generate_keypoints(matcher, image_a: numpy.ndarray, image_b: numpy.ndarray, device: str) -> tuple[
    numpy.ndarray, numpy.ndarray]:
    tensor_a = frame_to_tensor(image_a, device)
    tensor_b = frame_to_tensor(image_b, device)

    prediction: MatchingOutput = matcher(MatchingInput(image0=tensor_a, image1=tensor_b))
    kpts0, kpts1 = prediction.keypoints0[0].cpu().numpy(), prediction.keypoints1[0].cpu().numpy()
    matches, confidence = prediction.matches0[0].cpu().numpy(), prediction.matching_scores0[0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    return mkpts0, mkpts1


@capture_timing_info()
def combine_images_naive(previous_image, next_image, matrix, next_image_offset, current_size):
    h, w = next_image.shape
    next_image_offset.x -= matrix[0][2]
    next_image_offset.y -= matrix[1][2]
    previous_image_offset = Point()

    if next_image_offset.x < 0:
        current_size.x += abs(next_image_offset.x)
        previous_image_offset.x = -next_image_offset.x
        next_image_offset.x = 0

    if next_image_offset.y < 0:
        current_size.y += abs(next_image_offset.y)
        previous_image_offset.y = -next_image_offset.y
        next_image_offset.y = 0

    if next_image_offset.x + w > current_size.x:
        current_size.x = next_image_offset.x + w
    if next_image_offset.y + h > current_size.y:
        current_size.y = next_image_offset.y + h

    new_image = numpy.zeros((int(current_size.y), int(current_size.x)), numpy.uint8)

    slice_for_previous_image_y = slice(int(previous_image_offset.y),
                                       int(previous_image_offset.y + previous_image.shape[0]))
    slice_for_previous_image_x = slice(int(previous_image_offset.x),
                                       int(previous_image_offset.x + previous_image.shape[1]))
    slice_for_next_image_y = slice(int(next_image_offset.y), int(next_image_offset.y + next_image.shape[0]))
    slice_for_next_image_x = slice(int(next_image_offset.x), int(next_image_offset.x + next_image.shape[1]))

    new_image[slice_for_previous_image_y, slice_for_previous_image_x] = previous_image
    new_image[slice_for_next_image_y, slice_for_next_image_x] = next_image

    return new_image


@capture_timing_info()
def do_stitching_naive(config: Config):
    if not os.path.exists(config.output_directory):
        os.makedirs(config.output_directory)

    image_paths = [join(config.input_directory, filename) for filename in listdir(config.input_directory) if
                   splitext(filename)[1][1:] in config.supported_file_extensions]
    image_paths.sort()
    images = list(map(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), image_paths))

    image_pairs = list(zip(images[:-1], images[1:]))
    image_path_pairs = list(zip(image_paths[:-1], image_paths[1:]))

    matcher = Matching(SuperPointConfig(nms_radius=config.nms_radius, keypoint_threshold=config.keypoint_threshold,
                                        max_keypoints=config.max_keypoints),
                       SuperGlueConfig(weights=config.superglue_weights, sinkhorn_iterations=config.sinkhorn_iterations,
                                       match_threshold=config.match_threshold)).eval().to(config.device)

    current_size = Point(images[0].shape[1], images[0].shape[0])
    next_image_offset = Point(0.0, 0.0)
    previous_image = images[0]

    for i, (image_a, image_b) in enumerate(image_pairs):

        mkpts0, mkpts1 = generate_keypoints(matcher, image_a, image_b, config.device)

        if not mkpts0.shape[0] > 0 or not mkpts1.shape[0] > 0:
            print(
                f"[INFO]: No Keypoints extracted in iteration {i} when matching {basename(image_path_pairs[i][0])} with {basename(image_path_pairs[i][1])}!")
            print(f"[INFO]: Continuing!")
            previous_image = image_b
            next_image_offset = Point()
            current_size = Point(previous_image.shape[1], previous_image.shape[0])
            continue

        matrix, _ = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
        if matrix is None:
            print(
                f"[INFO]: Unable to estimate transform between {basename(image_path_pairs[i][0])} and {basename(image_path_pairs[i][1])} on iteration {i} ! ")
            print(f"[INFO]: Continuing!")
            previous_image = image_b
            next_image_offset = Point()
            current_size = Point(previous_image.shape[1], previous_image.shape[0])
            continue

        previous_image = combine_images_naive(previous_image, image_b, matrix, next_image_offset, current_size)
        cv2.imwrite(join(config.output_directory, f"intermediate_{i}.jpg"), previous_image)
