import numpy
from src.util.metrics import capture_timing_info


@capture_timing_info()
def calculate_profile(images: list[numpy.ndarray], smooth_steps: int = 0) -> numpy.ndarray:
    result = images[0].astype(float)
    for image in images[1:]:
        result += image.astype(float)
    result /= len(images)
    # TODO(Leah): Add Smoothing
    result /= numpy.amax(result)
    return result


@capture_timing_info()
def apply_profile_to_image(image: numpy.ndarray, profile: numpy.ndarray) -> numpy.ndarray:
    return numpy.clip(image / profile, 0, 255)
