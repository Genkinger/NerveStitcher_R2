import os.path

from offline_stitching import OfflineStitcher
from os.path import join, split
import torch

a = torch.load(join(split(__file__)[0], "models", "weights", "superglue_indoor.pth"))
print(a)
# stitcher = OfflineStitcher()

# TODO(Leah): Does this potential API make any sense?
# SuperPoint parts
# stitcher.set_input_directory("./")
# stitcher.set_output_directory("./Results")
# stitcher.set_working_title("EG7_A3")
# stitcher.load_images()
# stitcher.preprocess_images()
# stitcher.compute_raw_interest_points_and_descriptors()
# stitcher.save_raw_interest_points_and_descriptors("./Results/superpoint_checkpoint.pickle")
# stitcher.load_raw_interest_points_and_descriptors("./Results/superpoint_checkpoint.pickle")
# stitcher.filter_interest_points_and_descriptors()

# SuperGlue parts
# stitcher.set_matching_list([(1,2),(2,3)...])
# stitcher.compute_raw_matching_scores()
# stitcher.save_raw_matching_scores("./Results/superglue_checkpoint.pickle")
# stitcher.load_raw_matching_scores("./Results/superglue_checkpoint.pickle")
# stitcher.filter_matching_scores(callable)
# stitcher.predict_artefacts()
# stitcher.correct_artefacts()
# stitcher.drop_artefacts()
# stitcher.compose_final_image()
