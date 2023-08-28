from dataclasses import dataclass, field
import torch


@dataclass
class SuperPointConfiguration:
    nms_radius: int = 4
    keypoint_threshold: float = 0.005
    max_keypoints: int = 1024
    descriptor_dimensions: int = 256
    border_pixel_exclude_pixel_radius: int = 4

@dataclass
class SuperGlueConfiguration:
    superglue_weights: str = "indoor"
    sinkhorn_iterations: int = 100
    match_threshold: float = 0.80

@dataclass
class Configuration:
    force_cpu: bool = False
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    supported_file_extensions: list[str] = field(default_factory=lambda: ["jpg", "tif", "png"])
    interest_point_data_file_extension: str = "nsi"
    match_data_file_extension: str = "nsm"
    superpoint: SuperPointConfiguration = field(default_factory=lambda:SuperPointConfiguration())
    superglue: SuperGlueConfiguration = field(default_factory=lambda:SuperGlueConfiguration())

global_configuration = Configuration()