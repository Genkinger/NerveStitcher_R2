from os import listdir


def make_stitching_list(image_directory_path, sorting_key=None):
    images = [filename for filename in listdir(image_directory_path) if
              filename.endswith(".jpg") or filename.endswith(".tif")]
    images.sort(key=sorting_key)
    zipped_images = list(zip(images[:-1], images[1:]))
    return list(zipped_images)
