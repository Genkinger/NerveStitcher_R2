import numpy
from src.util.metrics import capture_timing_info


@capture_timing_info()
def calculate_brightness_and_contrast(images: list[numpy.ndarray]) -> tuple[float, float, list[tuple[float, float]]]:
    result = []
    series_brightness = 0
    series_contrast = 0
    for image in images:
        brightness = image.mean()
        contrast = numpy.sqrt((image ** 2).mean())
        series_brightness += brightness
        series_contrast += contrast
        result.append((brightness, contrast))
    series_brightness /= len(images)
    series_contrast /= len(images)
    return series_brightness, series_contrast, result


@capture_timing_info()
def apply_brightness_correction_to_image(image: numpy.ndarray, image_brightness: float, image_contrast: float, series_brightness: float, series_contrast: float) -> numpy.ndarray:
    return ((255.0 - series_brightness) / (255.0 - image_brightness)) * ((series_contrast * (image - image_brightness)) / image_contrast) + series_brightness

