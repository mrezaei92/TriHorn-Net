# TriHorn-Net
This repository contains the PyTorch implementation of TriHorn-Net. It contains easy instruction to replicate the results reported in the paper.  
arXiv preprint can be found at: https://X.com

### Introduction

We propose a novel network arcitecture that we call TriHorn-Net. It consists of two stages. In the first stage, the input depth image is run through the encoder network f. The encoder extracts and combines low-level and high-level features of the hand and outputs a high resolution feature volume, which is passed on to three separate branches. The UV branch, computes a per-joint attention map, where each map is focused on pixels where the corresponding joint occurs. This behavior is explicitly enforced by the application of 2D supervision to the heatmaps computed by passing the attention maps through a special softmax layer. The second branch, called the attention enhancement branch, also computes a per-joint attention map but does so under no constraints, allowing it to freely learn to detect the hand pixels most important for estimating the joint depth values under different scenarios. This attention map enhances the attention map computed by the UV branch through a fusion operation, which is performed by a linear interpolation controlled by per-joint learnable parameters. As a result, the fused attention maps attend to not only the joint pixels but also the hand pixels that do not belong to joints but contain useful information for estimating the joint depth values. The fused attention map is then used as guidance for pooling features from the depth feature map computed by the depth branch. Finally, a weight-sharing linear layer is used to estimate the joint depth values from the feature vectors computed for each joint.


![](https://drive.google.com/uc?export=view&id=13i7XQKINhHbJiNCiJjuSdhL_hF3SOVeW)

<div align=center> Fig 1. TriHorn-Net overview.</div>



## Setup
Download the repository:
```bash
makeReposit = [/the/directory/as/you/wish]
mkdir -p $makeReposit/; cd $makeReposit/
git clone https:https://github.com/mrezaei92/infrustructure_HPE.git
```
## Preparing the Dataset
1. NYU dataset
   
   Download and extract the dataset from the link provided below
   
   Copy the content of the folder data/NYU to where the dataset is located
   
   
2. ICVL dataset
   
   Download the file test.pickle from [here](https://drive.google.com/file/d/1cdTTDsJREZQC9ggVgF_2D7ZmFVVc2Hyk/view?usp=sharing)
   
   Download and extract the training set from the link provided below
   
   Navigate to the folder data/ICVL. Run the following command to get a file named train.pickle:  
   ``` python prepareICVL_train.py ICVLpath/Training```  
   Here, ICVLpath represents the address where the training set is extracted
   
   Place both test.pickle and train.pickle in one folder. This folder will serve as the ICVL dataset folder


3. MSRA dataset
  
   Download and extract the dataset from the link provided below
   
   Download and extract data/MSRA.tar.xz and copy its content to where the dataset is located 


## Training and Evaluation

Before running the experiment, first set the value ”datasetpath” in the corresponding .yaml file located in the folder configs. This value should be set to the address of the corresponding dataset. Then open a terminal and run the corresponding command.
After running each command, training is first done, and then the resulting models will be evaluated on the corresponding test set.  
The results will be saved in a file named ”results.txt”.

1. NYU

   ```bash
   bash train_eval_NYU.bash
   ```
  

2. ICVL

   ```bash
   bash train_eval_ICVL.bash
   ```

3. MSRA

   ```bash
   bash train_eval_MSRA.bash
   ```
   

## Supported Datasets
This repo supports using the following dataset for training and testing:

* ICVL Hand Poseture Dataset [[link](https://labicvl.github.io/hand.html)] [[paper](http://www.iis.ee.ic.ac.uk/dtang/cvpr_14.pdf)]
* NYU Hand Pose Dataset [[link](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm)] [[paper](https://cims.nyu.edu/~tompson/others/TOG_2014_paper_PREPRINT.pdf)]
* MSRA Hand Pose Dataset [[link](https://jimmysuen.github.io/)] [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf)]


## Results
The table below shows the predicted labels on ICVL, NYU and MSRA dataset. All labels are in the format of (u, v, d) where u and v are pixel coordinates.

| Dataset | Predicted Labels |
|-------|-------|
| ICVL | [Download](https://drive.google.com/file/d/1QqZbQS8wqxxahbmOgQHBWK59lGl7zXnU/view?usp=sharing) | 
| NYU | [Download](https://drive.google.com/file/d/11wLja_Xvu6knqdIctd_fpM3aYeeLtuSc/view?usp=sharing)|
| MSRA | [Download](https://drive.google.com/file/d/1T5nN_CK9qD5y1iSCapt2oyuMugLi4buQ/view?usp=sharing) | 




## Bibtex
If you use this paper for your research or projects, please cite [XX](https://dl.acm.org/doi).

```bibtex
@inproceedings{rezaei2021weakly,
  title={Weakly-supervised hand part segmentation from depth images},
  author={Rezaei, Mohammad and Farahanipad, Farnaz and Dillhoff, Alex and Elmasri, Ramez and Athitsos, Vassilis},
  booktitle={The 14th PErvasive Technologies Related to Assistive Environments Conference},
  pages={218--225},
  year={2021}
}
