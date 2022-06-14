import torch
import os
import numpy as np
from dataloader import *
import logging
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import time

################ GETTERs ############################

def get_optimizer(optimizer_type, model, args):

    no_decay = [] if args.decay_bias else ['bias', 'bn']
    grouped_parameters = [ 
        {'params': [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if optimizer_type=="adam":
        optimizer = torch.optim.Adam(grouped_parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    elif optimizer_type=="sgd":
        optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr,momentum=args.momentum,nesterov=args.nesterov)
    
    return optimizer


def get_scheduler(optimizer, args):

    scheduler_type=args.scheduler
        
    if scheduler_type=="steplr": 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.decay_step,gamma=args.learning_rate_decay)
    elif scheduler_type=="cosineWarmap":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T0, T_mult=args.Tmult, eta_min=args.eta_min)
    elif scheduler_type=="cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.num_epoch, T_mult=args.Tmult, eta_min=args.eta_min)
    elif scheduler_type=="auto":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=args.patience, min_lr=1e-8, factor = args.factor_scheduler,threshold =args.threshold_scheduler)

    return scheduler


def get_lossFunction(lossFunc_type):
    if lossFunc_type=="L2":
        lossFunction = torch.nn.MSELoss(reduction='none')

    elif lossFunc_type=="L1":
        lossFunction = torch.nn.L1Loss(reduction='none')

    elif lossFunc_type=='huber':
        from utils.torch_utils import My_SmoothL1Loss
        lossFunction = My_SmoothL1Loss().cuda()
    else:
        raise NotImplementedError

    return lossFunction




def DATA_Getters(args):
    
    cubic_size=[args.cubic_size,args.cubic_size,args.cubic_size]
    image_size=(args.cropSize,args.cropSize)

    datase_length = DATASET_LENGTHS[args.dataset]
    
    if args.subsetLength==-1: #full dataset
        labeled_subset=None
    else:
        np.random.seed(args.randseed)
        subsetLength = np.int32( np.round(args.subsetLength/100 * datase_length) )
        labeled_subset = np.random.choice([i for i in range(datase_length)], subsetLength,replace=False)
        unlabeled_subset = list( set([i for i in range(datase_length)])-set(labeled_subset) )
        
        e=time.time()
        e=int((e-int(e))*1000000)
        np.random.seed(e)
    
    base_parameters=dict(basepath=args.datasetpath, train=True, cropSize=image_size, 
            doJitterRotation=args.RotAugment, doAddWhiteNoise=args.doAddWhiteNoise, sigmaNoise=args.sigmaNoise,
            rotationAngleRange=[-45.0, 45.0], comJitter=args.comjitter, RandDotPercentage=args.RandDotPercentage,
            indeces=labeled_subset, cropSize3D=cubic_size, do_norm_zero_one=False, 
            random_seed=args.randseed,drop_joint_num=args.drop_joint_num,center_refined=args.center_refined,
            horizontal_flip=args.horizontal_flip, scale_aug=args.scale_aug)


    if args.dataset=="nyu":
        print("NYU dataset will be used")
        specs=dict(doLoadRealSample=(args.dataset_type=="real"),camID=args.camid,basepath=os.environ.get('NYU_PATH', args.datasetpath))
        base_parameters.update(specs)
        labled_train=NYUHandPoseDataset(**base_parameters)
        
    elif args.dataset=="icvl":
        print("ICVL dataset will be used")
        specs=dict(basepath= os.environ.get('ICVL_PATH', args.datasetpath))
        base_parameters.update(specs)
        labled_train=ICVLHandPoseDataset(**base_parameters)

        
    elif args.dataset=="msra":
        print("MSRA dataset will be used")
        specs=dict(basepath= os.environ.get('MSRA_PATH', args.datasetpath), LeaveOut_subject=args.leaveout_subject , use_default_cube=args.use_default_cube)
        base_parameters.update(specs)
        labled_train=MSRAHandPoseDataset(**base_parameters)


    unlabeled_train = None

    print("Total number of labeled samples to use for Training: %d"%(len(labled_train)))
    
    
    return labled_train, unlabeled_train



################################################################################


def loss_masked(preds,target,mask,lossFunc):

    #preds, targets: both tensor of shape (B,num_joint,k)
    # mask a tensor of shape (B,num_joint,1)

    joint_dim=preds.shape[2]
    distance = lossFunc(preds,target) # B,num_joints,k
    distance=distance*mask

    num_noneZero = torch.sum(mask)*joint_dim
    if num_noneZero == 0:
        num_noneZero=1
    
    return torch.sum(distance)/num_noneZero


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])




def Normalize_depth(preds,sizes,coms,add_com=False):
    # preds is a tensor of shape (B,k,3)
    # sizes is a tensor of shape (B,3)
    #coms is a tensor of shape (B,3)
    # this function denormalizes the depths of the prediction
    
    preds[:,:,2]=preds[:,:,2]*sizes[:,2][...,None]/2 #scale back
    
    if add_com:
        preds[:,:,2]=preds[:,:,2]+coms[:,2][...,None]
        
    return preds




def print_tensor(x):
    # x should be a one-dimensional tensor
    s=''
    for i in range(x.shape[0]):
        s=s+f"{i}|{x[i]:.3f} "
        
    return s





def tensor_to_dict(x):
    assert x.squeeze().ndim == 1 ,"The input tensor should be one dimensional"
    output_dict = {str(i):x[i].item() for i in range(len(x.squeeze()))}
    return output_dict

class AverageMeter(object):
    def __init__(self, fmt=':.3f'):
        self.data = {}
        self.counts = {}
        self.averages = {}
        self.fmt = fmt

    def reset(self):
        self.data = {}
        self.counts = {}
        self.averages = {}
        
    def update(self,data:dict):
        for key in data:
            self.data[key] = self.data.get(key,0) + data[key].detach()#(data[key].item() if torch.is_tensor(data[key]) else data[key])
            self.counts[key] = self.counts.get(key,0) + 1
            self.averages[key] = self.data[key] / self.counts[key]

    def get_dict(self):
        # output = {}
        # for key in self.data: 
        #     avg = self.averages[key]#self.data[key] / self.counts[key]
        #     if avg.numel() ==1:
        #         output[key] = avg
        return self.averages


    def synchronize_between_processes(self, average=True):
        # average values across all nodes
        if not is_dist_avail_and_initialized():
            return

        self.averages = reduce_dict(self.averages, average=average)


    def __str__(self):
        fmtstr = ''
        for key in self.data: 
            avg = self.averages[key]#self.data[key] / self.counts[key]
            if avg.numel()==1:
                avg = avg.item()
                fmtstr = fmtstr + f'{key.split("?")[0]}: ' + ('{avg' + self.fmt + '}').format(avg=avg)+ ", "
            else:
                avg =  print_tensor(avg.flatten())
                fmtstr = fmtstr + f'\n{key.split("?")[0]}: '+ avg + "\n"
            
            
            
        return fmtstr#fmtstr[2:]


def Grad_Updater(loss, model, optimizer, fp16_scaler, args):

    optimizer.zero_grad()
            
    if fp16_scaler is None:
        loss.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()
    else:
        fp16_scaler.scale(loss).backward()
        if args.clip_max_norm > 0:
            fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        fp16_scaler.step(optimizer)
        fp16_scaler.update()

def make_checkpoint(chk_name, model, optimizer, scheduler, args):
    data={"model":(model.module.state_dict() if not args.paralelization_type=="N" else model.state_dict()) , "args":args, "scheduler":scheduler.state_dict() }#"optimizer":optimizer.state_dict(), 
    torch.save(data, os.path.join(args.checkpoints_dir, "checkpoints", chk_name ))
    
def make_checkpoint2(epoch_num, model_dict, optimizer, scheduler, args):
    data={"args":args, "scheduler":scheduler.state_dict() }#"optimizer":optimizer.state_dict(), 
    for name,model in model_dict.item():
        data[name] = (model.module.state_dict() if not args.paralelization_type=="N" else model.state_dict())
    
    chk_name = "savedModel_E{}.pt".format(epoch_num)
    torch.save(data, os.path.join(args.checkpoints_dir, "checkpoints", chk_name ))
    

def load_checkpoint(model, args , optimizer, scheduler, device):
    data=torch.load(args.model_path, map_location=f"cuda:{device}")
    if hasattr(model, 'module'):
        model.module.load_state_dict(data["model"])
    else:
        model.load_state_dict(data["model"])
    
    #scheduler.load_state_dict(data["scheduler"])
    #optimizer.load_state_dict(data["optimizer"])
    print("Checkpoint loaded");
    
    
def getLogger(save_path = None, name = "Main", level = "INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s] %(message)s\n')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    log_file_path="log.txt" if save_path is None else os.path.join(save_path, 'log.txt')
    fileHandler = logging.FileHandler(log_file_path)
    fileHandler.setFormatter(log_format)
    logger.addHandler(fileHandler)
    
    return logger


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            assert hasattr(args,k),f"The yaml file does not have the atribute: {k}"
            setattr(args, k, dic[k])

          



def model_builder(model_name,num_joints, args):
    #assert model_name in ["resnet18","resnet50","hglass1","hglass2", "hglass1_small", "hglass2_small"]

    arc_name = model_name.split("_")[0].lower()

    
    
    if arc_name == "hglass":
        from model_factory.hourglass import HourglassNet, Bottleneck

        num_blocks = int(model_name.split("_")[1].lower())
        num_stack = int(model_name.split("_")[2].lower())

        return HourglassNet(Bottleneck, num_stacks=num_stack, num_blocks=num_blocks, num_classes=num_joints,BN=True,num_G=16, train_spread=args.train_spread)

    
    raise NotImplementedError



########## Distributed #########
# Mostly copy-paste from https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/utils.py, with some slight modifications

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values_to_reduce = input_dict[k]
            dist.all_reduce(values_to_reduce)
            values.append( values_to_reduce / (world_size if average else 1))

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict
