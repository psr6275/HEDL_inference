import argparse
import logging
import os
import torch.nn as nn

from crypten_utils import MultiProcessLauncher
import crypten.communicator as comm
import crypten
import torch

parser = argparse.ArgumentParser(description="CrypTen Cifar Training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "-b",
    "--batch-size",
    default=10000,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128)",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)

parser.add_argument("--tau", default=0.5, type=float, help="threshold parameter for the combined network")

parser.add_argument("--nu", default=100.0, type=float, help="smoothing parameter for the combined network (sigmoid)")

parser.add_argument(
    "--skip-plaintext",
    default=False,
    action="store_true",
    help="Skip validation for plaintext network",
)

parser.add_argument(
    "--sigmoid-combine",
    default=False,
    action="store_true",
    help="Sigmoid combine or multiplex combine",
)


parser.add_argument(
    "--multiprocess",
    default=True,
    action="store_true",
    help="Run example in multiprocess mode",
)

def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from crypten_utils import run_mpc_cifar

    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    
    run_mpc_combpart(
        args.batch_size,
        args.seed,
        args.tau,
        args.nu,
        args.sigmoid_combine,
        args.skip_plaintext,
    )
    print("="*10)
    print("total communication stats")
    comm.get().print_communication_stats()


def run_mpc_combpart(batch_size, seed, tau, nu, sigmoid_combine, skip_plaintext):  
    import random
    import torch
    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    xinput = torch.rand((batch_size, 16*5*5))*2-1
    youtput = torch.rand((batch_size, ))
    net = SimpleNet()
    
    if not skip_plaintext:
        logging.info("===== Evaluating plaintext combined SimpleNet network =====")
        condin_ori = evaluate_combpart(xinput,youtput, net, tau, nu, sigmoid_combine)
    logging.info("===== Evaluating Private combined SimpleNet network =====")
    crypten.print_communication_stats()
    print("*"*30)
    input_size = xinput.shape
    net_enc = construct_private_net(input_size, net)
    condin = evaluate_combpart(xinput,youtput, net_enc, tau, nu, sigmoid_combine, True)

from crypten_utils import encrypt_data_tensor_with_src

def evaluate_combpart(xinput, youtput, net, tau, nu, sigmoid_combine, priv=False):
    net.eval()
    if priv:
        x_enc = encrypt_data_tensor_with_src(xinput)
        y_enc = encrypt_data_tensor_with_src(youtput)
    else:
        x_enc = xinput
        y_enc = youtput
        
    output = net(x_enc)
    max_val = output.max(dim=1)[0]
    if sigmoid_combine:
        cond_in = (nu*(tau-max_val)).sigmoid()
#         return cond_in*y_enc + (1-cond_in)*y_enc
    else:
        cond_in = max_val>tau
        if priv==False:
#             print("non-private",priv)
            cond_in = cond_in.float()
#         else:
#             print("private")
    return cond_in.view(-1,1)*y_enc + (1-cond_in.view(-1,1))*y_enc
    

def construct_private_net(input_size, model):
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)
    
    if rank==0:
        model_upd = model
    else:
        model_upd = SimpleNet()
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model

def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)

class SimpleNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = self.fc(x)
        return x        

if __name__ == "__main__":
    main(_run_experiment)