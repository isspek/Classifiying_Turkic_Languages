import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Analysing Turkic Languages')
    parser.add_argument("--mode", type=str)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--fname", type=str)
    parser.add_argument("--type", type=str, choices=["all", "experiments"])
    parser.add_argument("--force", action='store_true',
                        help='model will be trained even it is trained before')

    args = parser.parse_args()
    return args
