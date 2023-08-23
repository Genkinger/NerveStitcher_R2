from offline_stitching import OfflineStitcher
import sys
from matplotlib import pyplot as plt


def visualize(stitcher, index):
    image, scores, descriptors = stitcher.interst_point_data[index]

    fig, (a,b,c) = plt.subplots(1,3)
    a.imshow(image,cmap="greys")
    b.imshow(scores.squeeze(),cmap="greys")
    c.imshow(descriptors.squeeze()[0,:,:])
    fig.show()


stitcher = OfflineStitcher()

# TODO(Leah): Does this potential API make any sense?
# SuperPoint parts
stitcher.set_input_directory(sys.argv[1])
stitcher.set_output_directory(sys.argv[2])
stitcher.set_working_title("A3")
stitcher.load_images()
#stitcher.preprocess_images()
stitcher.compute_raw_interest_points_and_descriptors()
stitcher.save_raw_interest_points_and_descriptors("./points.pickle")

visualize(stitcher,99)

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
