#! /usr/bin/env python

import yaml
import argparse

arg = argparse.ArgumentParser("DDPM Parameters")
arg.add_argument("-c", "--config", type=str, default="config.yml", help="Param config files")

if __name__ == "__main__":
    opt = arg.parse_args()
    with open(opt.config, "r") as fin:
        conf = yaml.SafeLoader(fin)