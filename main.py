import argparse
import json
import os
from datetime import datetime

from torch.backends import cudnn

from solver import Solver
from data_loader import get_loader


def save_config(config, shared_path):
    save_path = os.path.join(shared_path, "params.json")
    with open(save_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)    


def get_time():
    return datetime.now().strftime("%d%m_%H%M%S")


def main(config):
    cudnn.benchmark = True
    config.time_now = get_time()

    # ---- run folder: ./experiments/<run_name>_<time>/ (same layout as the reference repo)
    run_name = config.run_name.replace(" ", "_").replace("/", "_")
    folder_name = "{}_{}".format(run_name, config.time_now) if run_name else config.time_now
    shared_path = os.path.join(config.base_path, folder_name)

    config.model_path = os.path.join(shared_path, "models")
    config.logs_path = os.path.join(shared_path, "logs")
    config.results_path = os.path.join(shared_path, "results")
    for p in (config.model_path, config.logs_path, config.results_path):
        os.makedirs(p, exist_ok=True)
    save_config(config, shared_path)
    print("Run folder:", shared_path)

    # ---- data: built in memory, three different seeds so the sets never overlap
    print("Generating data...")
    train_loader = get_loader(batch_size=config.batch_size, mode='train',
                              num_workers=config.num_workers,
                              n_per_snr=config.n_per_snr, h=config.h,
                              max_cfo=config.cfo, seed=config.data_seed)
    val_loader = get_loader(batch_size=256, mode='test',
                            num_workers=config.num_workers,
                            n_per_snr=config.val_n_per_snr, h=config.h,
                            max_cfo=config.cfo, seed=config.data_seed + 2)
    test_loader = get_loader(batch_size=256, mode='test',
                             num_workers=config.num_workers,
                             n_per_snr=config.test_n_per_snr, h=config.h,
                             max_cfo=config.cfo, seed=config.data_seed + 1)

    solver = Solver(config, train_loader, val_loader, test_loader)
    solver.train()   # trains, saves best.pth / last.pth, then tests best.pth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_classes', type=int, default=2)    # 0 = noise only, 1 = signal present
    parser.add_argument('--seed', type=int, default=0)           # torch seed (weight init, shuffling)
    parser.add_argument('--num_workers', type=int, default=0)

    # data (passed to data_loader.get_loader)
    parser.add_argument('--n_per_snr', type=int, default=500)       # training examples per SNR per class
    parser.add_argument('--val_n_per_snr', type=int, default=100)
    parser.add_argument('--test_n_per_snr', type=int, default=500)
    parser.add_argument('--h', type=float, default=0.5)             # BFSK modulation index
    parser.add_argument('--cfo', type=float, default=0.0)           # max carrier frequency offset
    parser.add_argument('--data_seed', type=int, default=0)         # train=seed, test=seed+1, val=seed+2

    # misc
    parser.add_argument('--base_path', type=str, default='./experiments/')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--log_step', type=int, default=50)

    config = parser.parse_args()
    print(config)
    main(config)