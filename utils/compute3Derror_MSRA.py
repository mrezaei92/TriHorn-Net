import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import sys
import os

def prediction_fileToArray(address,num_joints):
    # the input file should a file where each line stores a hand in the form of: joint1_1 joint1_2 joint1_3 joint2_1 ....
    inputfile = open(address)
    inputfile.seek(0)
    lines=inputfile.readlines()
    all_joints=[]
    for i in range(len(lines)):
        s=lines[i]
        part = s.split(' ')
        #     print(s)
        gtorig = np.zeros((num_joints, 3), np.float32)
        for joint in range(num_joints):
            for xyz in range(0, 3):
                gtorig[joint, xyz] = part[joint*3+xyz]

        uvd=gtorig
        all_joints.append(uvd)

    all_joints=np.array(all_joints)
    inputfile.close()
    return all_joints



def convert_uvd_to_xyz_tensor(uvd ):
        # uvd is a tensor of  size(B,num_joints,3)
        fx, fy, ux , uy = (241.42, 241.42, 160., 120.)
        xyz = torch.zeros(uvd.shape);
        xyz[:,:,2] = uvd[:,:,2];
        xyz[:,:,0] = (uvd[:,:,0]-ux)*uvd[:,:,2]/fx
        xyz[:,:,1] = (uvd[:,:,1]-uy)*uvd[:,:,2]/fy
        return xyz

    

pred_file = sys.argv[1]

preds = prediction_fileToArray(pred_file,21)
gts = prediction_fileToArray(os.path.join(os.environ['MSRA_PATH'],"msra_test_groundtruth_label.txt"),21)

#print(preds.shape,gts.shape)
errs=torch.norm(convert_uvd_to_xyz_tensor(torch.from_numpy(preds))-convert_uvd_to_xyz_tensor(torch.from_numpy(gts)),dim=-1)

print(f"The 3D error is : {torch.mean(errs)}")
