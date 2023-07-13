from dataclasses import dataclass, field
import cv2
import torch
from os import makedirs, listdir
from os.path import join, exists

from src.models.matching import Matching
from src.models.utils import read_image
from src.legacy.combine_img_legacy import combine

torch.set_grad_enabled(False)


def make_stitching_list(image_directory_path, sorting_key=None):
    images = [filename for filename in listdir(image_directory_path) if
              filename.endswith(".jpg") or filename.endswith(".tif")]
    images.sort(key=sorting_key)
    zipped_images = list(zip(images[:-1], images[1:]))
    return list(zipped_images)


@dataclass
class StitcherConfig:
    input_image_directory: str = "./data/test/stitch_img2"
    final_stitch_output_directory: str = join(input_image_directory, "match")
    nms_radius: int = 4
    keypoint_threshold: float = 0.005
    max_keypoints: int = 1024
    superglue_weights: str = "indoor"
    sinkhorn_iterations: int = 100
    match_threshold: float = 0.80
    resize: list[float] = field(default_factory=lambda: [-1, -1])
    resize_float: bool = False
    force_cpu: bool = False
    intermediate_stitch_output_directory: str = join(input_image_directory, "result")
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"


def do_stitching(stitcher_config: StitcherConfig):
    current_stitcher_config = stitcher_config

    if not exists(current_stitcher_config.intermediate_stitch_output_directory):
        makedirs(current_stitcher_config.intermediate_stitch_output_directory)

    if not exists(current_stitcher_config.final_stitch_output_directory):
        makedirs(current_stitcher_config.final_stitch_output_directory)

    image_pairs = make_stitching_list(current_stitcher_config.input_image_directory,
                                      sorting_key=lambda x: int(x[4:-4]))

    first_stitching_image = cv2.imread(join(current_stitcher_config.input_image_directory, image_pairs[0][0]))

    if len(current_stitcher_config.resize) == 2 and current_stitcher_config.resize[1] == -1:
        resize = current_stitcher_config.resize[0:1]

    matching_config = {
        "superpoint": {
            "nms_radius": current_stitcher_config.nms_radius,
            "keypoint_threshold": current_stitcher_config.keypoint_threshold,
            "max_keypoints": current_stitcher_config.max_keypoints
        },
        "superglue": {
            "weights": current_stitcher_config.superglue_weights,
            "sinkhorn_iterations": current_stitcher_config.sinkhorn_iterations,
            "match_threshold": current_stitcher_config.match_threshold,
        }
    }

    matching = Matching(matching_config).eval().to(current_stitcher_config.device)

    print(f"input directory： \"{current_stitcher_config.input_image_directory}\"")
    print(f"output directory： \"{current_stitcher_config.final_stitch_output_directory}\"")
    print(f"running on device： \"{current_stitcher_config.device}\"")
    print("Starting...")

    x_move, y_move = [], []
    count = 0  # Current Stitch Counter

    ### STITCHING LOOP
    for i, current_pair in enumerate(image_pairs):
        print("Stitching", current_pair[0], current_pair[1])

        image0, inp0, scales0 = read_image(
            join(current_stitcher_config.input_image_directory, current_pair[0]),
            current_stitcher_config.device, resize, 0, current_stitcher_config.resize_float)

        image1, inp1, scales1 = read_image(
            join(current_stitcher_config.input_image_directory, current_pair[1]),
            current_stitcher_config.device, resize, 0, current_stitcher_config.resize_float)

        if image0 is None or image1 is None:
            print("Unable to load one of the images: {} {}, aborting".format(
                join(current_stitcher_config.input_image_directory, current_pair[0]),
                join(current_stitcher_config.input_image_directory, current_pair[1])))
            exit(1)

        pred = matching({"image0": inp0, "image1": inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches, conf = pred["matches0"], pred["matching_scores0"]

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        try:
            H, SS = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
            assert H is not None
        except:
            if x_move == [] and y_move == []:
                print(f"{current_pair[0]} is an isolated image，unable to splice！")
                first_stitching_image = image1
                count += 1
                print(current_pair[1])
                print("starting with next batch")
                continue
            else:
                print(len(x_move))
                count, result_img = combine(first_stitching_image, x_move, y_move,
                                            current_stitcher_config.input_image_directory, image_pairs, count,
                                            current_stitcher_config.intermediate_stitch_output_directory)
                cv2.imwrite(join(current_stitcher_config.final_stitch_output_directory, f"result{i}.jpg"), result_img)
                first_stitching_image = image1
                print(current_pair[1])
                x_move, y_move = [], []
                print("starting with next batch")
        else:
            x_move.append(H[0, 2])
            y_move.append(H[1, 2])

    if x_move != [] and y_move != []:
        count, result_img = combine(first_stitching_image, x_move, y_move,
                                    current_stitcher_config.input_image_directory,
                                    image_pairs,
                                    count, current_stitcher_config.intermediate_stitch_output_directory)
        cv2.imwrite(join(current_stitcher_config.final_stitch_output_directory, "result.jpg"), result_img)
        cv2.imwrite(join(current_stitcher_config.intermediate_stitch_output_directory, "result.jpg"), result_img)

    print(f"Spliced {count} Images")
    print("Done！")
