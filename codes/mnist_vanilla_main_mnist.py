import argparse
import logging
import os

from crypten_utils import MultiProcessLauncher
import crypten.communicator as comm

parser = argparse.ArgumentParser(description="CrypTen MNIST Training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128)",
)

parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)

parser.add_argument(
    "--net-location",
    default="../results/mnist_orig_mpc.pth",
    type=str,
    metavar="PATH",
    help="path to real model checkpoint (default: none)",
)

parser.add_argument(
    "--data-location",
    default="../data/mnist",
    type=str,
    metavar="PATH",
    help="path to fake model checkpoint (default: none)",
)

parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)

# parser.add_argument("--tau", default=0.5, type=float, help="threshold parameter for the combined network")

parser.add_argument(
    "--skip-plaintext",
    default=False,
    action="store_true",
    help="Skip validation for plaintext network",
)

parser.add_argument(
    "--multiprocess",
    default=True,
    action="store_true",
    help="Run example in multiprocess mode",
)

def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from crypten_utils import run_mpc_mnist_vanilla

    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    
    run_mpc_mnist_vanilla(
        args.batch_size,
        args.net_location,
        args.data_location,
        args.seed,
        args.skip_plaintext,
        args.print_freq,
    )
    print("="*10)
    print("total communication stats")
    comm.get().print_communication_stats()


    
    
def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)