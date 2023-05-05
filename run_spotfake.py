from faknow.run.run_spotfake import run_spotfake
import argparse
import torch.multiprocessing as mp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-d", type=str, default="/root/FaKnow/dataset/example/SpotFake/twitter/",
        help="root of datasets"
    )
    parser.add_argument(
        "--bertname", "-bn", type=str, default="bert-base-uncased", help="pretrained bert name"
    )
    args, _ = parser.parse_known_args()
    mp.spawn(
        run_spotfake,
        args=(args.dataset, args.bertname)
    )
