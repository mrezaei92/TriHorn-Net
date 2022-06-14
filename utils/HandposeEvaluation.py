
import numpy as np
import torch
import os
from enum import Enum
import cv2


class HandposeEvaluation(object):
 
    def __init__(self, predictions, gt):
        # both predictions and gt should be np.array of shape (batch,num_key,3)
        self.gt=gt
        self.joints=predictions
    def getMeanError(self):
        """
        get average error over all joints, averaged over sequence
        :return: mean error
        """
        return np.nanmean(np.nanmean(np.sqrt(np.square(self.gt - self.joints).sum(axis=2)), axis=1))
    
    def GetMeanErrorPerJoint(self):
        d=np.abs(self.gt-self.joints)
        distances=np.linalg.norm(d,axis=2)
        return np.mean(distances,axis=0)
    
    def getNumFramesWithinMaxDist(self, dist):
        """
        calculate the number of frames where the maximum difference of a joint is within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (np.nanmax(np.sqrt(np.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()

      
    def getErrorPerDimension(self,printOut=True):
        d=np.abs(self.gt-self.joints)
        xmean=torch.mean(d[:,:,0])
        ymean=torch.mean(d[:,:,1])
        zmean=torch.mean(d[:,:,2])
        msg="mean along X direction = %.2f \nmean along Y direction = %.2f\nmean along Z direction = %.2f"%(xmean,ymean,zmean)
        if printOut:
            print(msg)
        
        return msg

       
    def update(self, predictions, gt):
        self.gt=gt
        self.joints=predictions

       
       
  
  
######################################## Visualization tools ################################################ adopted and changed from https://github.com/RenFeiTemp/SRN/blob/master/realtime/vis_tool.py

def get_sketch_setting(dataset):
    if dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15)]
    elif dataset == 'nyu':
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]
    elif dataset == 'nyu_full':
        return [(20,3),(3,2),(2,1),(1,0),(20,7),(7,6),(6,5),(5,4),(20,11),(11,10),(10,9),(9,8),(20,15),(15,14),(14,13),(13,12),(20,19),(19,18),(18,17),(17,16),
               (20,21),(20,22)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    elif dataset == 'hands17':
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11), (3, 12), (12, 13), (13, 14), (4, 15), (15, 16),
                (16, 17), (5, 18), (18, 19), (19, 20)]


class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (17, 240, 244)#(204, 153, 17) #
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    BROWN = (204, 153, 17)


def get_sketch_color(dataset):
    if dataset == 'icvl':
        return [Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.BLUE, Color.GREEN,
                Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.CYAN, Color.CYAN)
    elif dataset == 'nyu_full':
        return (Color.GREEN,Color.GREEN,Color.GREEN,Color.GREEN, Color.RED, Color.RED, Color.RED, Color.RED,  Color.PURPLE, Color.PURPLE,Color.PURPLE,Color.PURPLE,
                Color.YELLOW,Color.YELLOW,Color.YELLOW,Color.YELLOW,
                Color.BLUE, Color.BLUE,  Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED,
              Color.GREEN, Color.GREEN, Color.GREEN,
              Color.BLUE, Color.BLUE, Color.BLUE,
              Color.YELLOW, Color.YELLOW, Color.YELLOW,
              Color.PURPLE, Color.PURPLE, Color.PURPLE,
              Color.RED, Color.RED, Color.RED]



def get_joint_color(dataset):
    if dataset == 'icvl':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'nyu_full':
        return (Color.GREEN, Color.GREEN,Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.RED, Color.RED,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.CYAN, Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.PURPLE, Color.PURPLE, Color.PURPLE,
                Color.RED, Color.RED, Color.RED]


def draw_pose(img, pose, dataset="nyu"):
    # img is a tensor of shape (H,W) its values are in [-1,1] background is 1
    # pose is tensor or np.array of shape (k,d)
    
    colors = get_sketch_color(dataset)
    colors_joint = get_joint_color(dataset)
    img=np.array(255*np.array((img+1)/2))
    img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, colors_joint[idx].value, thickness=-1)
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 1)
        idx = idx + 1
        
    
    
    return img

   
   
   
   
   
   ########################## File I/O ######################################
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
   
def prediction_ArrayToFile(preds,address):
    np.savetxt(address, preds.reshape(preds.shape[0],-1), fmt="%.3f")
    
    
   
   
   
   
   
   
   
   
   
