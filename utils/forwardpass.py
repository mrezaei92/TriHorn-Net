import torch
import os
import numpy as np
import time

from utils.utils import  Normalize_depth, loss_masked

FORWARD_PASS = {'hglass': "normal"}

EVAL_FUNCTIONS = {'hglass': "normal"}

################ GETTERs ############################

def get_forwardPass(train_mode,args):
    arc_name = args.model_name.split("_")[0].lower()
    mode = FORWARD_PASS[arc_name]
    if mode == "normal":
        return Normal_forwardPass

    else:
        raise NotImplementedError


def get_EvalFunction(args):
    arc_name = args.model_name.split("_")[0].lower()
    mode = EVAL_FUNCTIONS[arc_name]
    if mode == "normal":
        return Normal_eval
    else:
        raise NotImplementedError

######################### Forward passes #######################################


def Normal_forwardPass(model, inputs, gt_uvd, lossFunction, cubesize,coms, joint_mask, visible_mask, args, suffix = None):
    mask = joint_mask * visible_mask
    output = model(inputs)
    output = Normalize_depth(output,sizes=cubesize,coms=coms,add_com=False)# # torch.mean(abs(output-gt_uvd)*mask[...,None])
    loss = loss_masked(output, gt_uvd, mask, lossFunction)#torch.mean( lossFunction(output,gt_uvd)*mask )
    loss_dict = {}
    suffix = ('' if suffix is None else suffix)
    loss_dict[f"{suffix}Tot_loss"] = loss.detach()
    loss_dict["visibility_rate"] = (torch.sum(visible_mask,dim=0)/visible_mask.shape[0]).squeeze()

    return loss, loss_dict
    return loss, loss_dict



############################ Eval Functions ###################################################
def Normal_eval(inputs,outputs,cubesize,com,args):
    outputs[:,:,0:2]=outputs[:,:,0:2]
    return Normalize_depth(outputs,cubesize,com,add_com=True)

