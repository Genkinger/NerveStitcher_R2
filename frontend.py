import os
import sys

from stitching import do_stitching, StitcherConfig
from vignetting_correction import do_vignetting_correction
import argparse


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

    def stitch(self):
        parser = argparse.ArgumentParser(
            description="Runs the final stitching on the given input directory"
        )
        parser.add_argument("-i", "--input_directory", required=True)
        parser.add_argument("-o", "--final_stitch_directory", required=True)
        parser.add_argument("-s", "--intermediate_stitch_directory", required=True)
        parser.add_argument("-m", "--matching_threshold", required=False, default=0.80)
        parser.add_argument("-k", "--keypoint_threshold", required=False, default=0.005)
        parser.add_argument("-n", "--nms_radius", required=False, default=4)

        arguments = parser.parse_args(sys.argv[2:])

        stitcher_config = StitcherConfig(input_image_directory=arguments.input_directory,
                                         final_stitch_output_directory=arguments.final_stitch_directory,
                                         intermediate_stitch_output_directory=arguments.intermediate_stitch_directory,
                                         match_threshold=arguments.matching_threshold,
                                         nms_radius=arguments.nms_radius,
                                         keypoint_threshold=arguments.keypoint_threshold)
        do_stitching(stitcher_config)


Frontend()
