from os.path import join, splitext
from os import listdir

def get_file_paths_with_extensions(directory_path,extensions):
    return [join(directory_path,filename) for filename in listdir(directory_path) if splitext(filename)[1][1:] in extensions]