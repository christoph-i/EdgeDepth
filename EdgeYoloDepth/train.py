from edgeyolo import launch as train
import argparse


def make_parser():
    parser = argparse.ArgumentParser("EdgeYOLO train parser")
    parser.add_argument("-c", "--cfg", type=str, default="/.../.../train_params.yaml")

    # Not commend
    parser.add_argument("--default", action="store_true", help="use default train settings in edgeyolo/train/default.yaml")
    return parser.parse_args()


if __name__ == '__main__':
    args = make_parser()
    train("DEFAULT" if args.default else args.cfg)
