import argparse
import os

def get_args_parser():
    parser = argparse.ArgumentParser(description='training')

    # Dataset
    parser.add_argument('--model_name',default="hglass_2_1",type=str,help="hglass_numBlock_numStack")
    
    parser.add_argument('--dataset',default="nyu", choices=('nyu', 'icvl','msra'),type=str,help="which dataset to use")

    parser.add_argument('--num_workers',default=4, type=int,help="how many subprocesses to use for data loading")
    
    parser.add_argument('--camid',default=1, choices=(1,2,3),type=int,help="which view to load samples from")

    parser.add_argument('--cubic_size', default=280, type=int, help="The size of the cubic around the hand")

    parser.add_argument('--cropSize', default=128, type=int, help="The size of input depth image")

    parser.add_argument('--datasetpath', default=os.environ.get('NYU_PATH', '/data/NYU') , type=str, help="the address of the dataset")

    parser.add_argument('--randseed',default=32, type=int,help="the seed for generating a random subset of data")

    parser.add_argument('--subsetLength',default=-1, type=int,help="the size (in percentage) of the random subset of data (if -1, use the entire dataset)")
    
    parser.add_argument('--drop_joint_num',default=0, type=int,help="The number of joints that are randomly dropped for each frame")

    parser.add_argument('--dataset_type',default="real", choices=('real', 'synthetic'),type=str,help="wether to use real or synthetic dataset")
    
    ##### for MSRA      
    parser.add_argument('--leaveout_subject', default=1, type=int, help="The subject to be left out for Val/Test")
    parser.add_argument('--use_default_cube', default=1, type=int, help="Whether to use default 3D cubic size for each subject")
    
    
    ## Data Augmentations
    parser.add_argument('--center_refined',default=1, type=int, help="wether use refined COMs from a file")

    parser.add_argument('--RotAugment',default=1, type=int, help="wether to do rotation augmentation")
    
    parser.add_argument('--doAddWhiteNoise',default=0, type=int, help="wether to do noise augmentation")

    parser.add_argument('--sigmaNoise',default=1, type=float,help="standard deviation of noise augmentation")

    parser.add_argument('--comjitter',default=8, type=float, help="the degree of adding random noise to COM")
    
    parser.add_argument('--RandDotPercentage',default=0, type=float,help="an augmentation where it randomly zeros out a percentage of foreground pixels")
    
    parser.add_argument('--horizontal_flip',default=0, type=int,help="wether to apply random horizontal flip ")
    
    parser.add_argument('--scale_aug',default=1, type=int, help="use scale data augmentation from the range [0.8-1.2]")
    


    # General settings
    
    parser.add_argument('--paralelization_type',default="N", choices=('DP', 'DDP',"N"), type=str,help="DPP: DistributedDataParallel, DP: DataParallel, N: single-GPU")

    parser.add_argument('--device_IDs',default="0,1,2,3",type=str,help="the id of GPUs to use, e.g. 2,3,1")

    parser.add_argument('--default_cuda_id',default="0",type=str)

    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')

    parser.add_argument('--ngpus_per_node',default="4",type=str,help="Num of GPUs that each Node has")

    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')

    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,  help='url used to set up distributed training')

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

    parser.add_argument('--use_fp16', type=int, default=0, help="""Whether or not to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling mixed precision if the loss is unstable, or if each GPU device is not operating at full capacity""")
    
    parser.add_argument('--clip_max_norm', default=0, type=float, help='gradient clipping max norm')
    
    parser.add_argument('--use_tensorboard',default=1, type=int, help="whether to use tensorboard to log")

    parser.add_argument('--use_logger',default=1, type=int, help="whether to use logger")

    parser.add_argument('--save_freq',default=10, type=int, help="the frequency with which checkpoints are made")
    

    parser.add_argument('--epoch',default=100,dest="num_epoch",type=int)

    parser.add_argument('--batch_size',default=32,type=int,dest="batch_size")

    parser.add_argument('--model_path',type=str, help="the address where the model is saved for resuming the training")

    parser.add_argument('--checkpoints_dir',type=str, help="where to save model chekpoints", required=True)    
    
    parser.add_argument('--LossFunction',type=str,default="L1", choices=('L2', 'L1','huber'), help="The choice of the loss function")

    parser.add_argument('--train_mode',type=str,default="normal", choices=('normal', 'aleatoric_hg', 'aleatoric_resnet'), help="The choice of the loss function, NOTE: when using aleatoric, L2 lossFunc should be used")
    
    parser.add_argument('--joint_dim',default=3,type=int,help="wether to do 3D or 2D pose estimation")
    
    parser.add_argument('--decay_bias',default=1, type=int, help="whether to use decay on bias weights")

    parser.add_argument('--config_file',default="",type=str,help="whether or not to load .yaml config file to overwrite some args arguments")
    
    # Optimizer

    parser.add_argument('--optimizer',type=str,default="adam", choices=('adam', 'sgd'), help="The choice optimizer")

    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--weight_decay', default=2e-5, type=float)

    parser.add_argument('--momentum', default=0.9, type=float, help="the momentum used for SGD optimizer")

    parser.add_argument('--nesterov', default=0, type=int, help="whether to use nesterov for SGD optimizer")


    #### Scheduler

    parser.add_argument('--scheduler',type=str,default="cosine", choices=('steplr', 'cosine','cosineWarmap','auto'), help="The choice for scheduler")


    ##### for stepLR	    
    parser.add_argument('--learning_rate_decay', default=1e-1, type=float, help="when steplr is used")
    
    parser.add_argument('--decay_step', default=10, type=int, help="The learning rate will be multiplied by learning_rate_decay every decay_step epochs")


    ##### for cosineWarmap	    
    parser.add_argument('--T0', default=10, type=int, help="when cosine is used: the first number of epochs for restarting. if T0=num_epochs -> cosineAnnealing scheduler")
    
    parser.add_argument('--Tmult', default=2, type=float, help="when cosine is used: after each restart, T is multiplied by Tmult")

    parser.add_argument('--eta_min', default=0.0, type=float, help="when cosine is used: the minimum learning rate to reach")


    ##### for Auto      
    parser.add_argument('--patience', default=10, type=int, help=" Number of epochs with no improvement after which learning rate will be reduced")
    
    parser.add_argument('--factor_scheduler', default=0.1, type=float, help="Factor by which the learning rate will be reduced. new_lr = lr * factor")

    parser.add_argument('--threshold_scheduler', default=1e-1, type=float, help="Threshold for measuring the new optimum, to only focus on significant changes")


    parser.add_argument('--train_spread', default=0, type=int, help="whether to train the Temprature used in spatial softmax for the output heatmaps")


    

    return parser
    


# configuration for different training scenarios:    
'''
Single-GPU training:

--paralelization_type N --default_cuda_id X

where X is the cuda ID of the GPU to use
===================================================================

Multi-GPU training using DataParallel (multi-gpus on one machine):

--paralelization_type DP --default_cuda_id X --device_IDs U

where X is the cuda ID of the default GPU to use, and U is the GPU IDs to use, e.g. 3,2,4
====================================================================

Multi-GPU training using DistributedDataParallel (multi-gpus on one machine): (faster than DP)

--paralelization_type DDP --device_IDs U --ngpus_per_node X --batch C --rank 0 --dist-url tcp://127.0.0.1:29500 

where  U is the GPU IDs to use, e.g. 3,2,4
X: the number of GPUs to use
C: batch size, the batch places on each GPU will be of size X/C
====================================================================
Multi-GPU training using DistributedDataParallel (multi-gpus on more than one machine): 

for the first meachine, run this: 

--paralelization_type DDP --device_IDs U --ngpus_per_node X,Y,Z... --batch C --rank R 

Then for other machines run:
--paralelization_type DDP --device_IDs U --ngpus_per_node X,Y,Z... --batch C --rank R --dist-url ADD 


U: is the GPU IDs to use on each machine, e.g. 3,2,4

X,Y,Z ... : denote the number of GPUs to be used on each machine respectively, for are the same for all commands across the machines

C: denotes the batch size, which is devided by the number of GPUs for each machine. For example, for the first machine, the batch size on each GPU will be C/X, for the second machine X/Y and so on.

R denotes the rank of the machine withing that cluster of machine, it is 0 for the first machine, 1 for the second machine and so on

ADD: the address to look at for the settings for the processes coordination, e.g. 'tcp://127.0.0.1:23456'


'''
# **IMPORTANT NOTE about Batch_size: in DP, the effective batch size is batch_size devided by the number of GPUs, as it evenly splits the input data over the GPUs
# at each iteration. So batch statistics are computed on each device seprately. In case of DDP, if we use batch_norm_sync, then the effective batch size will be
# batch_size as it will be synchronized across GPUs. If we don't use batch_norm_sync, then the effective batch size will be batch_size devided by the number of GPUs
