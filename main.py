import torch
import numpy as np
import builtins

import torch.optim as optim
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import logging


from config import *
from utils.utils import *
from engine import Train


def main(args):
        
    if os.path.exists(args.checkpoints_dir):
        print("checkpoint dir already exists")
    else:
        os.mkdir(args.checkpoints_dir)
        os.mkdir(os.path.join(args.checkpoints_dir, "checkpoints")) # create the folder for saving the checkpoints
        print("checkpoint dir created")
     

    ngpus_per_node = [int(i) for i in args.ngpus_per_node.split(",")]
    current_node_GPU_counts=ngpus_per_node[args.rank]
    
    if args.paralelization_type=="DDP":
        args.world_size = np.sum(ngpus_per_node)
        mp.spawn(main_worker, nprocs=current_node_GPU_counts, args=(ngpus_per_node, args, current_node_GPU_counts))
    else:
        main_worker(int(args.default_cuda_id), ngpus_per_node, args , current_node_GPU_counts)

      

def main_worker(gpu, ngpus_per_node, args, current_node_GPU_counts):
    
    ########################## Model ##########################
    
    rank=-1
   
    model= model_builder(args.model_name,num_joints=DATASET_NUM_JOINTS[args.dataset], args = args)
    
    device_IDs=[int(i) for i in args.device_IDs.split(",")]
        
    default_cuda_id = "cuda:{}".format(args.default_cuda_id)

    
    if args.paralelization_type=="DDP":
        assert len(device_IDs)==current_node_GPU_counts
        
        ngpus_per_node_padded=[0]+ngpus_per_node
        rank = np.sum(ngpus_per_node_padded[:args.rank+1]) + gpu
        
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=rank)
        torch.distributed.barrier()
        
        print("All processes joined, ready to start!")
        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        torch.cuda.set_device("cuda:{}".format(device_IDs[gpu]))
        model.cuda(device_IDs[gpu])
        
        args.batch_size = int(args.batch_size / current_node_GPU_counts)
        #args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_IDs[gpu]])
        
        device = device_IDs[gpu]

        
    elif args.paralelization_type=="DP":
        device = torch.device(default_cuda_id)
        model=model.to(device)
        model=torch.nn.DataParallel(model,device_ids=device_IDs)

    
    elif args.paralelization_type=="N":
        device = torch.device(default_cuda_id)
        torch.cuda.set_device(device)
        model = model.cuda()

    
    # supress print if it is not the master process       
    if not is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        if args.use_logger:
            print("Logger will be used!")
            logger = getLogger(save_path = None, name = "Main", level = "INFO")
            builtins.print =  logger.info
    
    print("\n"+"##"*15 + "\n" + str(args) + "\n\n" + "##"*15 + "\n")
        
    print(f" World_size =  {get_world_size()} !!!")

    if args.clip_max_norm > 0:
        print("Gradient Clipping will be used")
        

    ########################## Dataset and Optimizer ##########################

    data_loaders = {}

    labled_train, unlabeled_train = DATA_Getters(args)
    
    labeled_sampler = torch.utils.data.distributed.DistributedSampler(labled_train) if args.paralelization_type=="DDP" else None
        
    data_loaders["trainloader_labeled"] = torch.utils.data.DataLoader( labled_train, batch_size=args.batch_size,
               shuffle=(labeled_sampler is None), num_workers=args.num_workers, pin_memory=True,
               sampler=labeled_sampler, drop_last=True)

  
    data_loaders["trainloader_unlabeled"] = None

    
    optimizer = get_optimizer(args.optimizer, model, args)

    scheduler = get_scheduler(optimizer, args)
        
    lossFunction = get_lossFunction(args.LossFunction)
    
    torch.backends.cudnn.benchmark = True

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print("fp16_scaler being used!")
        

    if args.model_path is not None:
        load_checkpoint(model, args , optimizer, scheduler, device)
    

    print(f"Model to be trained: {args.model_name}")
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

########################## Main Loop ##########################
    Train(model, data_loaders, args,lossFunction,optimizer,device,scheduler, fp16_scaler, rank)

    
         
    print('Finished Training')

####################################

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    over_write_args_from_file(args, args.config_file)
    main(args)	
    
    
    
    
