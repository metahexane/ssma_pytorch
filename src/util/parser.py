import argparse

def parse_args():
    """
    Parses the inputs of the program
    :return:
    """
    parser = argparse.ArgumentParser(description="SSMA for semantic segmentation")

    parser.add_argument("--data", nargs='?', default="data/RAND_CITYSCAPES/",
                        help="The path to the data folder")
    parser.add_argument("--enc-lr", nargs='?', default='[0.001, 0.0001, 0]',
                        help="The learning rate for the encoder per stage")
    parser.add_argument("--dec-lr", nargs='?', default='[0.001, 0.001, 0.00001]',
                        help="The learning rate for the decoder, eASPP and SSMA blocks per stage")
    parser.add_argument("--iters", nargs='?', default='[150000, 100000, 50000]',
                        help="The amount of iterations per stage")
    parser.add_argument("--batch", nargs='?', default='[8, 12, 7]',
                        help="The batch size per stage")
    parser.add_argument("--eval", nargs='?', default='0',
                        help="Whether to evaluate the model after each epoch")
    parser.add_argument("--save", nargs='?', default='1',
                        help="Whether to save the model")
    parser.add_argument("--save-checkpoint", nargs='?', default='5000',
                        help="Saves each x amount of iterations")
    parser.add_argument("--start", nargs='?', default='1',
                        help="The starting stage, the model parameters must also be specified if set so another value than 1")
    parser.add_argument("--model", nargs="?", default="",
                        help="The model date/signature")
    parser.add_argument("--resume", nargs="?", default="0",
                        help="Whether to continue training or to start training all over again from the stage given by the start parameter")

    return parser.parse_args()