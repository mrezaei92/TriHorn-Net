# TriHorNet
This is the PyTorch implementation of TriHornNet

arXiv preprint can be found at: https://X.com


## Setup
Download the repository:
```bash
makeReposit = [/the/directory/as/you/wish]
mkdir -p $makeReposit/; cd $makeReposit/
git clone https:https://github.com/mrezaei92/infrustructure_HPE.git
```
# Preparing the Dataset
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


# Training and Evaluation

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
   

# Supported Datasets
This repo supports using the following dataset for training and testing:

* ICVL Hand Poseture Dataset [[link](https://labicvl.github.io/hand.html)] [[paper](http://www.iis.ee.ic.ac.uk/dtang/cvpr_14.pdf)]
* NYU Hand Pose Dataset [[link](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm)] [[paper](https://cims.nyu.edu/~tompson/others/TOG_2014_paper_PREPRINT.pdf)]
* MSRA Hand Pose Dataset [[link](https://jimmysuen.github.io/)] [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf)]


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
