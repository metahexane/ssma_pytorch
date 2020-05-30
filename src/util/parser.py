import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SSMA for semantic segmentation")

    parser.add_argument("--data", nargs='?', default="data/RAND_CITYSCAPES/")
    parser.add_argument("--enc-lr", nargs='?', default='[0.001, 0.0001, 0]')
    parser.add_argument("--dec-lr", nargs='?', default='[0.001, 0.001, 0.00001]')
    parser.add_argument("--iters", nargs='?', default='[150000, 100000, 50000]')
    parser.add_argument("--batch", nargs='?', default='[8, 7, 12]')
    parser.add_argument("--eval", nargs='?', default='0')
    parser.add_argument("--save", nargs='?', default='1')

    return parser.parse_args()