from offline_stitching import OfflineStitcher
import sys
from matplotlib import pyplot as plt
import torch
from util.metrics import print_metrics

# def visualize(stitcher, index):
#     image, scores, descriptors = stitcher.interest_point_data[index]
#     plt.imshow(image,cmap="Greys_r")
#     keypoints = torch.nonzero(torch.from_numpy(scores.squeeze()) > 0.005)
#     keypoints = keypoints.transpose(0,1)
#     plt.scatter(keypoints[1].numpy(),keypoints[0].numpy(),s=1,c="Red")
#     plt.show()


stitcher = OfflineStitcher()

# TODO(Leah): Does this potential API make any sense?
# SuperPoint parts
stitcher.set_input_directory(sys.argv[1])
stitcher.set_output_directory(sys.argv[2])
stitcher.set_working_title("A3")
stitcher.load_images()
stitcher.preprocess_images()
stitcher.compute_raw_interest_points_and_descriptors(f"{sys.argv[2]}/raw_ip.pkl")
#stitcher.save_raw_interest_points_and_descriptors("./points.pickle")

print_metrics(True)

#stitcher.load_raw_interest_points_and_descriptors("./points.pickle")
#stitcher.save_raw_interest_points_and_descriptors("./Results/superpoint_checkpoint.pickle")
#stitcher.load_raw_interest_points_and_descriptors("./Results/superpoint_checkpoint.pickle")
#stitcher.filter_interest_points_and_descriptors()

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
