import os
import torch

from disnet import DisNet, DisNetSingleClass, DisNetClassIds, DisNetSingleClassVehicleBottom, DisNetClassIdsInclXY



out_dir = r"/.../..."


torch.set_num_threads(8)

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



def init_distributed(args) -> None:
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.rank = int(os.environ.get('RANK', 0))

    args.distributed = args.world_size > 1

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        torch.cuda.set_device(f'cuda:{args.rank}')

    assert args.rank >= 0

############################################
###########   Params #######################
############################################

args = Args()

# Hardcoded necessary params
args.dataset = 'kitti'

args.num_gpus = 0

args.lr = 1e-4 # default = 1e-4
args.device = "cpu"
args.optimizer = "adam"  # choices=("adam", "sgd")
args.max_patience = 100 # default 20
args.test_only = False  # Test only without training
args.accumulation_steps = 4 # default = 4 (number of mini batches accumulated before gradient updates)
args.epochs = 1500 # default = 1000 (as suggested in paper)
args.batch_size = 2 # default = 2
args.weight_decay = 1e-5 # default = 1e-5
args.n_workers = 0 # default = 4
args.scheduler = "cosine" # default = "cosine" | choices=("cosine", "plateau", "none")

args.wandb = False
args.long_range = False
args.use_debug_dataset = False


# single class mode
args.single_class_active_class = "vehicle"

# hardcoded optional params

args.checkpoint = None # Path to checkpoint to load - if None train from scratch
args.resume = False
args.exp_name = None

if args.resume:
    assert (
            args.resume and args.exp_name is not None
    ), "Cannot resume without --exp_name"

# auto generated params

args.exp_log_path = out_dir

init_distributed(args)




############################################
########## Train and Test Model ############
############################################

def main(args) -> None:

    # model = DisNet(args)
    # model = DisNetSingleClass(args)
    # model = DisNetClassIds(args)
    # model = DisNetSingleClassVehicleBottom(args)
    model = DisNetClassIdsInclXY(args)

    disnet_trainer = model.get_trainer()(model, args=args)

    disnet_trainer.run()


if __name__ == "__main__":
    main(args)