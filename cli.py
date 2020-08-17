import argparse

parser = argparse.ArgumentParser(description='Analysing Turkic Languages')
parser.add_argument("--mode", type=str)
parser.add_argument("--random_seed", type=int)
parser.add_argument("--fname", type=str)

args = parser.parse_args()
