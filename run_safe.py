from faknow.run.run_safe import run_safe
import argparse
import torch.multiprocessing as mp
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-d", type=str, default=r"C:\Users\10749\Desktop\FaKnow\dataset\example\SAFE",
        help="root of datasets"
    )
    args, _ = parser.parse_known_args()
    mp.spawn(
        run_safe,
        args=(Path(args.dataset),)
    )
