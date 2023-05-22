import cv2
import torch
import torch.utils.data
import numpy
import scipy.io as scio
import os
from PIL import Image

import progressbar as pb
import numpy as np
from scipy import stats, ndimage
import pickle
import tqdm

from scipy import stats, ndimage
import sys

def calculateCoM(dpt,minDepth=0,maxDepth=500):
        """
        Calculate the center of mass
        :param dpt: depth image
        :return: (x,y,z) center of mass
        """

        dc = dpt.copy()
        dc[dc < minDepth] = 0
        dc[dc > maxDepth] = 0
        cc = ndimage.measurements.center_of_mass(dc > 0)
        num = np.count_nonzero(dc)
        com = np.array((cc[1]*num, cc[0]*num, dc.sum()), float)

        if num == 0:
            return np.array((0, 0, 0), float)
        else:
            return com/num
            
            
def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
Terminal Saved Output    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = numpy.dot(numpy.asarray(M).reshape((3, 3)), numpy.asarray([pt[0], pt[1], 1]))
    return numpy.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoints2D(pts, M):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param M: transformation matrix
    :return: transformed points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
    return ret
    
    
class HandDetector(object):
    """
    Detect hand based on simple heuristic, centered at Center of Mass
    """

    RESIZE_BILINEAR = 0
    RESIZE_CV2_NN = 1
    RESIZE_CV2_LINEAR = 2

    def __init__(self, dpt, fx, fy, importer=None, refineNet=None):
        """
        Constructor
        :param dpt: depth image
        :param fx: camera focal lenght
        :param fy: camera focal lenght
        """
        self.dpt = dpt
        self.maxDepth = min(1500, dpt.max())
        self.minDepth = max(10, dpt.min())
        # set values out of range to 0
        self.dpt[self.dpt > self.maxDepth] = 0.
        self.dpt[self.dpt < self.minDepth] = 0.
        # camera settings
        self.fx = fx
        self.fy = fy
        # Optional refinement of CoM
        self.refineNet = refineNet
        self.importer = importer
        # depth resize method
        self.resizeMethod = self.RESIZE_CV2_NN


    def calculateCoM(self, dpt):
        """
        Calculate the center of mass
        :param dpt: depth image
        :return: (x,y,z) center of mass
        """

        dc = dpt.copy()
        dc[dc < self.minDepth] = 0
        dc[dc > self.maxDepth] = 0
        cc = ndimage.measurements.center_of_mass(dc > 0)
        num = numpy.count_nonzero(dc)
        com = numpy.array((cc[1]*num, cc[0]*num, dc.sum()), numpy.float)

        if num == 0:
            return numpy.array((0, 0, 0), numpy.float)
        else:
            return com/num

    def checkImage(self, tol):
        """
        Check if there is some content in the image
        :param tol: tolerance
        :return:True if image is contentful, otherwise false
        """
        # print numpy.std(self.dpt)
        if numpy.std(self.dpt) < tol:
            return False
        else:
            return True

    def getNDValue(self):
        """
        Get value of not defined depth value distances
        :return:value of not defined depth value
        """
        if self.dpt[self.dpt < self.minDepth].shape[0] > self.dpt[self.dpt > self.maxDepth].shape[0]:
            return stats.mode(self.dpt[self.dpt < self.minDepth])[0][0]
        else:
            return stats.mode(self.dpt[self.dpt > self.maxDepth])[0][0]


    def comToBounds(self, com, size):
        """
        Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: xstart, xend, ystart, yend, zstart, zend
        """
        if numpy.isclose(com[2], 0.):
            print ("Warning: CoM ill-defined!")
            xstart = self.dpt.shape[0]//4
            xend = xstart + self.dpt.shape[0]//2
            ystart = self.dpt.shape[1]//4
            yend = ystart + self.dpt.shape[1]//2
            zstart = self.minDepth
            zend = self.maxDepth
        else:
            zstart = com[2] - size[2] / 2.
            zend = com[2] + size[2] / 2.
            xstart = int(numpy.floor((com[0] * com[2] / self.fx - size[0] / 2.) / com[2]*self.fx+0.5))
            xend = int(numpy.floor((com[0] * com[2] / self.fx + size[0] / 2.) / com[2]*self.fx+0.5))
            ystart = int(numpy.floor((com[1] * com[2] / self.fy - size[1] / 2.) / com[2]*self.fy+0.5))
            yend = int(numpy.floor((com[1] * com[2] / self.fy + size[1] / 2.) / com[2]*self.fy+0.5))
        return xstart, xend, ystart, yend, zstart, zend

    def comToTransform(self, com, size, dsize=(128, 128)):
        """
        Calculate affine transform from crop
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: affine transform
        """

        xstart, xend, ystart, yend, _, _ = self.comToBounds(com, size)

        trans = numpy.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            scale = numpy.eye(3) * dsize[0] / float(wb)
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            scale = numpy.eye(3) * dsize[1] / float(hb)
            sz = (wb * dsize[1] / hb, dsize[1])
        scale[2, 2] = 1

        xstart = int(numpy.floor(dsize[0] / 2. - sz[1] / 2.))
        ystart = int(numpy.floor(dsize[1] / 2. - sz[0] / 2.))
        off = numpy.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return numpy.dot(off, numpy.dot(scale, trans))

    def getCrop(self, dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True, background=0):
        """
        Crop patch from image
        :param dpt: depth image to crop from
        :param xstart: start x
        :param xend: end x
        :param ystart: start y
        :param yend: end y
        :param zstart: start z
        :param zend: end z
        :param thresh_z: threshold z values
        :return: cropped image
        """
        if len(dpt.shape) == 2:
            cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1])].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                           abs(yend)-min(yend, dpt.shape[0])),
                                          (abs(xstart)-max(xstart, 0),
                                           abs(xend)-min(xend, dpt.shape[1]))), mode='constant', constant_values=background)
        elif len(dpt.shape) == 3:
            cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1]), :].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                           abs(yend)-min(yend, dpt.shape[0])),
                                          (abs(xstart)-max(xstart, 0),
                                           abs(xend)-min(xend, dpt.shape[1])),
                                          (0, 0)), mode='constant', constant_values=background)
        else:
            raise NotImplementedError()

        if thresh_z is True:
            msk1 = numpy.logical_and(cropped < zstart, cropped != 0)
            msk2 = numpy.logical_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.  # backface is at 0, it is set later
        return cropped


    def resizeCrop(self, crop, sz):
        """
        Resize cropped image
        :param crop: crop
        :param sz: size
        :return: resized image
        """
        if self.resizeMethod == self.RESIZE_CV2_NN:
            rz = cv2.resize(crop, sz, interpolation=cv2.INTER_NEAREST)
        elif self.resizeMethod == self.RESIZE_BILINEAR:
            rz = self.bilinearResize(crop, sz, self.getNDValue())
        elif self.resizeMethod == self.RESIZE_CV2_LINEAR:
            rz = cv2.resize(crop, sz, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError("Unknown resize method!")
        return rz



    def cropArea3D(self, com=None, size=(250, 250, 250), dsize=(128, 128)):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        #print com, self.importer.jointImgTo3D(com)
        #import matplotlib.pyplot as plt
        #import matplotlib
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        if com is None:
            com = self.calculateCoM(self.dpt)

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)
        #print xstart-xend,ystart-yend


        # crop patch from source
        cropped = self.getCrop(self.dpt, xstart, xend, ystart, yend, zstart, zend)
   
        #############
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        # print com, sz, cropped.shape, xstart, xend, ystart, yend, hb, wb, zstart, zend
        trans = numpy.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if cropped.shape[0] > cropped.shape[1]:
            scale = numpy.eye(3) * sz[1] / float(cropped.shape[0])
        else:
            scale = numpy.eye(3) * sz[0] / float(cropped.shape[1])
        scale[2, 2] = 1

        # depth resize
        rz = self.resizeCrop(cropped, (np.int32(np.round(sz[0])),np.int32(np.round(sz[1]))))


        ret = numpy.ones(dsize, numpy.float32) * self.getNDValue()  # use background as filler

        xstart = int(numpy.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(numpy.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape, xstart, ystart
        off = numpy.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, numpy.dot(off, numpy.dot(scale, trans)), com



class DepthImporter(object):
    """
    provide baisc functionality to load depth data
    """

    def __init__(self, fx, fy, ux, uy):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy
        self.depth_map_size = (320, 240)
        self.refineNet = None
        self.crop_joint_idx = 0

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x4 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret


class ICVLImporter(DepthImporter):
    """
    provide functionality to load data from the ICVL dataset
    """

    def __init__(self, basepath, useCache=True,refineNet=None):
        """
        Constructor
        :param basepath: base path of the ICVL dataset
        :return:
        """

        super(ICVLImporter, self).__init__(241.42, 241.42, 160., 120.)  # see Qian et.al.

        self.depth_map_size = (320, 240)
        self.basepath = basepath
        self.numJoints = 16
        self.crop_joint_idx = 0
        self.refineNet = refineNet
        self.default_cubes = {'train': (250, 250, 250),
                              'test': (250, 250, 250),
                              'test_seq_2': (250, 250, 250)}
        self.sides = {'train': 'right', 'test': 'right', 'test_seq_2': 'right'}
        
    def loadDepthMap(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)  # open image

        assert len(img.getbands()) == 1  # ensure depth image
        imgdata = numpy.asarray(img, numpy.float32)

        return imgdata

    
    
    def loadSequence(self, seqName, subSeq=None, Nmax=float('inf'), shuffle=False, rng=None, docom=False, cube=None,hand=None):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. train
        :param subSeq: list of subsequence names, e.g. 0, 45, 122-5
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """

        if (subSeq is not None) and (not isinstance(subSeq, list)):
            raise TypeError("subSeq must be None or list")

        if cube is None:
            config = {'cube': self.default_cubes[seqName]}
        else:
            assert isinstance(cube, tuple)
            assert len(cube) == 3
            config = {'cube': cube}


        # Load the dataset
        objdir = '{}/Depth/'.format(self.basepath)
        trainlabels = '{}/labels.txt'.format(self.basepath)
        
        f=open("icvl_train_list.txt", "r")
        ll=f.readlines()
        for i in range(0,len(ll)-1):
            ll[i]=ll[i][:-1]
            
            
        inputfile = open(trainlabels)

        txt = 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=len(inputfile.readlines()), widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()
        inputfile.seek(0)

        data = []
        i = 0
        for line in inputfile:
            # early stop
            if len(data) >= Nmax:
                break

            part = line.split(' ')
            # check for subsequences and skip them if necessary
            
            if part[0] not in ll:
                continue
           
           
            dptFileName = '{}/{}'.format(objdir, part[0])

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!".format(dptFileName))
                i += 1
                continue
            dpt = self.loadDepthMap(dptFileName)

            # joints in image coordinates
            gtorig = numpy.zeros((self.numJoints, 3), numpy.float32)
            for joint in range(self.numJoints):
                for xyz in range(0, 3):
                    gtorig[joint, xyz] = part[joint*3+xyz+1]
 
            # normalized joints in 3D coordinates
            gt3Dorig = self.jointsImgTo3D(gtorig)

           
            data.append([dpt, gtorig, gt3Dorig, dptFileName])
            pbar.update(i)
            i += 1

        inputfile.close()
        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return data


#######################################################
adress=sys.argv[1]
dataset=ICVLImporter(basepath=adress)
data=dataset.loadSequence("train")


q1=[]
for d in tqdm.tqdm(data):
    q1.append(calculateCoM(d[0].copy()))
    
q1=np.stack(q1)


dd=(data,q1)

with open('train.pickle', 'wb') as f:
    pickle.dump(dd, f)






















