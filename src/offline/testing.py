from offline_stitching import OfflineStitcher
import sys
from matplotlib import pyplot as plt
from configuration import global_configuration
from code import interact

stitcher = OfflineStitcher()

# TODO(Leah): Does this potential API make any sense?
# SuperPoint parts
stitcher.set_input_directory("../Datasets/EGT7_001-A_3_snp")
stitcher.set_output_directory("../Datasets/EGT7_001-A_3_snp/Outputs")
stitcher.set_working_title("first_superpoint_tests_preprocessed")
#stitcher.load_superpoint()
#stitcher.load_images()
#stitcher.preprocess_images()
#stitcher.compute_interest_points_and_descriptors(0.001, 1024)
#stitcher.save_interest_point_data(f"data_0.001.{global_configuration.interest_point_data_file_extension}")
stitcher.load_interest_point_data(f"data_0.001.{global_configuration.interest_point_data_file_extension}")

print(__file__)
interact(local=locals())

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
