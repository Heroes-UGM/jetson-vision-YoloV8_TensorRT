from models import TRTModule, TRTProfilerV0  # isort:skip
import argparse
import time
import torch


def profile(args):
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    random_input = torch.randn(Engine.inp_info[0].shape, device=device)
    for i in range(10):
        a = time.time()
        _ = Engine(random_input)
        print(time.time() - a)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    profile(args)
