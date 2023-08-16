from dataclasses import dataclass, field
import torch


@dataclass
class Configuration:
    nms_radius: int = 4
    keypoint_threshold: float = 0.005
    max_keypoints: int = 1024
    superglue_weights: str = "indoor"
    sinkhorn_iterations: int = 100
    match_threshold: float = 0.80
    descriptor_dimensions: int = 256
    force_cpu: bool = False
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    supported_file_extensions: list[str] = field(default_factory=lambda: ["jpg", "tif"])


configuration = Configuration()

