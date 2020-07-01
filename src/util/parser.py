import argparse

def parse_args():
    """
    Parses the inputs of the program
    :return:
    """
    parser = argparse.ArgumentParser(description="SSMA for semantic segmentation")

    parser.add_argument("--data", nargs='?', default="data/RAND_CITYSCAPES/")
    parser.add_argument("--enc-lr", nargs='?', default='[0.001, 0.0001, 0]')
    parser.add_argument("--dec-lr", nargs='?', default='[0.001, 0.001, 0.00001]')
    parser.add_argument("--iters", nargs='?', default='[25000, 10000, 5000]')
    parser.add_argument("--batch", nargs='?', default='[8, 8, 8]')
    parser.add_argument("--eval", nargs='?', default='0')
    parser.add_argument("--save", nargs='?', default='1')
    parser.add_argument("--save-checkpoint", nargs='?', default='5000')
    parser.add_argument("--start", nargs='?', default='1')
    parser.add_argument("--model", nargs="?", default="")
    parser.add_argument("--resume", nargs="?", default="0")

    return parser.parse_args()