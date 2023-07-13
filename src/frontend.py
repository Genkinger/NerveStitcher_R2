import os
import sys
import argparse
from main import do_stitching_naive, Config
from preprocessing.vignetting_correction import do_vignetting_correction
from util.metrics import print_metrics


class Frontend(object):
    def __init__(self):
        main_parser = argparse.ArgumentParser(prog="NerveStitcher_R2", description="Frontend for NerveStitcher_R2")
        main_parser.add_argument("command")
        arguments = main_parser.parse_args(sys.argv[1:2])
        if not hasattr(self, arguments.command):
            main_parser.print_help()
            exit(1)

        getattr(self, arguments.command)()

    def vignetting_correction(self):
        parser = argparse.ArgumentParser(
            description="Runs Vignetting Correction on the images in the input_directory and stores the result in output_directory")
        parser.add_argument("-i", "--input_directory", required=True)
        parser.add_argument("-o", "--output_directory", required=True)
        parser.add_argument("-r", "--reference", required=True,
                            help="Path to a single reference image that is used to compute a reference histogram")
        arguments = parser.parse_args(sys.argv[2:])

        if not os.path.exists(arguments.output_directory):
            os.makedirs(arguments.output_directory)

        do_vignetting_correction(arguments.referece, arguments.input_directory, arguments.output_directory)
        print_metrics(True)

    def stitch(self):
        parser = argparse.ArgumentParser(
            description="Runs the final stitching on the given input directory"
        )
        parser.add_argument("-i", "--input_directory", required=True)
        parser.add_argument("-o", "--output_directory", required=True)
        parser.add_argument("-m", "--matching_threshold", required=False, default=0.80)
        parser.add_argument("-k", "--keypoint_threshold", required=False, default=0.005)
        parser.add_argument("-n", "--nms_radius", required=False, default=4)

        arguments = parser.parse_args(sys.argv[2:])

        stitcher_config = Config(input_directory=arguments.input_directory,
                                 output_directory=arguments.output_directory,
                                 match_threshold=float(arguments.matching_threshold),
                                 nms_radius=int(arguments.nms_radius),
                                 keypoint_threshold=float(arguments.keypoint_threshold))
        do_stitching_naive(stitcher_config)
        print_metrics(True)


Frontend()
