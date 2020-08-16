import argparse

parser = argparse.ArgumentParser(description='Analysing Turkic Languages')
parser.add_argument("--mode", type=str)
parser.add_argument("--random_seed", type=int)

args = parser.parse_args()
