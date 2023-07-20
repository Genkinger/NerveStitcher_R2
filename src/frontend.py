import os
import sys
import argparse
from main import do_stitching_naive, Config
from preprocessing.vignetting_correction import do_vignetting_correction
from util.metrics import print_metrics
from dataclasses import fields, MISSING


class Frontend(object):
    def __init__(self):
        main_parser = argparse.ArgumentParser(prog="NerveStitcher_R2", description="Frontend for NerveStitcher_R2")
        main_parser.add_argument("command")
        arguments = main_parser.parse_args(sys.argv[1:2])
        if not hasattr(self, arguments.command):
            main_parser.print_help()
            exit(1)

        getattr(self, arguments.command)()

    @staticmethod
    def create_arg_parser_from_dataclass(dataclass, **kwargs) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(**kwargs)
        for x in fields(dataclass):
            if x.default_factory is not MISSING:
                parser.add_argument(f"--{x.name}", default=x.default_factory(), type=x.type)
            else:
                parser.add_argument(f"--{x.name}", default=x.default, type=x.type)
        return parser

    def stitch(self):
        parser = self.create_arg_parser_from_dataclass(
            Config, description="Runs the final stitching on the given input directory")
        arguments = parser.parse_args(sys.argv[2:])

        do_stitching_naive(Config(**vars(arguments)))
        print_metrics(True)


Frontend()
