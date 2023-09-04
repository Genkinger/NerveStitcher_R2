from offline_stitching import OfflineStitcher
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from configuration import global_configuration
from code import interact
import torch
import numpy
import cv2
#stitcher = OfflineStitcher()


def visualize(a,b,count,method=0):
    
    image_a,scores_a,coordinates_a,descriptors_a = a
    image_b,scores_b,coordinates_b,descriptors_b = b
    cnt = min(count,len(coordinates_a))
    coordinates_a = coordinates_a.cpu().numpy()[:cnt]
    coordinates_b = coordinates_b.cpu().numpy()[:cnt]
    scores_a = scores_a.cpu().numpy()[:cnt]
    scores_b = scores_b.cpu().numpy()[:cnt]
    descriptors_a = descriptors_a.cpu().numpy()[:cnt]
    descriptors_b = descriptors_b.cpu().numpy()[:cnt]
    
    #NOTE(Leah): This is only for testing stuff, use SuperGlue for correct results
    def dot_product_method(desc_a,desc_b):
        closest_matches = []
        for i in range(len(desc_a)):
            closest_match_index = -1
            current_max = -100000
            for j in range(len(desc_b)):
                res = numpy.dot(desc_a[i],desc_b[j])
                if res > current_max:
                    current_max = res
                    closest_match_index = j
            closest_matches.append(closest_match_index)
        return closest_matches
    def difference_method(desc_a,desc_b):
        closest_matches = []
        for i in range(len(desc_a)):
            closest_match_index = -1
            current_min = 100000
            for j in range(len(desc_b)):
                res = numpy.sqrt(numpy.sum((desc_a[i] - desc_b[j])**2))
                if res < current_min:
                    current_min = res
                    closest_match_index = j
            closest_matches.append(closest_match_index)
        return closest_matches

    closest_matches = dot_product_method(descriptors_a,descriptors_b) if method == 0 else  difference_method(descriptors_a,descriptors_b)
    f, axes = plt.subplots(1,2)
    axes[0].imshow(image_a,cmap="Greys_r")
    axes[0].scatter(coordinates_a[:,1],coordinates_a[:,0],c="maroon",s=scores_a*10)
    axes[1].imshow(image_b,cmap="Greys_r")
    axes[1].scatter(coordinates_b[:,1],coordinates_b[:,0],c="maroon",s=scores_b*10)

    for i in range(len(closest_matches)):
        match_a = numpy.flip(coordinates_a[i])
        match_b = numpy.flip(coordinates_b[closest_matches[i]])
        cp = ConnectionPatch(xyA=match_a,xyB=match_b,coordsA=axes[0].transData,coordsB=axes[1].transData,color="red",linewidth=0.8)
        f.add_artist(cp)

    f.show()


# TODO(Leah): Does this potential API make any sense?
# SuperPoint parts
#stitcher.set_input_directory("../Datasets/EGT7_001-A_3_snp")
#stitcher.set_output_directory("../Datasets/EGT7_001-A_3_snp/Outputs")
#stitcher.set_working_title("first_superpoint_tests_preprocessed")
#stitcher.load_superpoint()
#stitcher.load_images()
#stitcher.preprocess_images()
#stitcher.compute_interest_points_and_descriptors(0.001, 1024)
#stitcher.save_interest_point_data(f"data_0.001.{global_configuration.interest_point_data_file_extension}")
#stitcher.load_interest_point_data(f"data_0.005.{global_configuration.interest_point_data_file_extension}")




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

stitcher = OfflineStitcher()
stitcher.load_superpoint()

def test_superpoint_image_pair_simple(stitcher: OfflineStitcher,image_file_path: str, angle_deg: float, scale: float, translation: tuple[float,float], keypoint_threshold: float = 0.005, max_keypoints: int = 1024, dot_distance: bool = False):
        
    image = cv2.imread(image_file_path,cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((height/2,width/2),angle_deg, scale)
    print(rotation_matrix)
    #rotation_matrix = numpy.array([[numpy.cos(theta),-numpy.sin(theta),0],[numpy.sin(theta),numpy.cos(theta),0]])
    warp_matrix = rotation_matrix + numpy.array([[0,0,translation[0]],[0,0,translation[1]]])
    image_transformed = cv2.warpAffine(image, warp_matrix, (height,width))

    #f,axes = plt.subplots(1,2)
    #axes[0].imshow(image)
    #axes[1].imshow(image_transformed)
    #f.show()
    
    stitcher.images = [image, image_transformed]
    stitcher.interest_point_data = []
    stitcher.compute_interest_points_and_descriptors(keypoint_threshold, max_keypoints)
    visualize(stitcher.interest_point_data[0], stitcher.interest_point_data[1],max_keypoints,method=0 if dot_distance else 1)
    return warp_matrix

warp_matrix = test_superpoint_image_pair_simple(stitcher,"../Datasets/EGT7_A_3/EGT7_001-A_3_00002.tif",20,1,(0,0),0.002)

def apply_matrix_to_keypoint_positions(position_matrix: numpy.ndarray, matrix: numpy.ndarray):
    position_matrix = numpy.flip(position_matrix,axis=1)
    position_matrix = numpy.insert(position_matrix,2,1,axis=1)
    position_matrix = position_matrix.transpose()
    result = numpy.matmul(matrix,position_matrix)
    result = result.transpose()
    result = numpy.flip(result,axis=1)
    return result

transformed_keypoints = apply_matrix_to_keypoint_positions(stitcher.interest_point_data[0][2].cpu().numpy(),warp_matrix)
interact(local=locals())
