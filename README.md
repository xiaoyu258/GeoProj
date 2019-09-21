# GeoProj

### [Project page](https://xiaoyu258.github.io/projects/geoproj) | [Paper](https://arxiv.org/abs/1909.03459)

The source code of Blind Geometric Distortion Correction on Images Through Deep Learning by Li et al, CVPR 2019. 

<img src='imgs/results.jpg' align="center" width=850> 

## Prerequisites
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Dataset Generation
In order to train the model using the provided code, the data needs to be generated in a certain manner. 

You can use any distortion-free images to generate the dataset. In this paper, we use [Places365-Standard dataset](http://places2.csail.mit.edu/download.html) at the resolution of 512\*512 as the original non-distorted images to generate the 256\*256 dataset.

Run the following command for dataset generation:
```bash
python data/dataset_generate.py [--sourcedir [PATH]] [--datasetdir [PATH]] 
                                [--trainnum [NUMBER]] [--testnum [NUMBER]]

--sourcedir           Path to original non-distorted images
--datasetdir          Path to the generated dataset
--trainnum            Number of generated training samples
--testnum             Number of generated testing samples
```

### Training
Run the following command for help message about optional arguments like learning rate, dataset directory, etc.
```bash
python trainNetS.py --h # if you want to train GeoNetS
python trainNetM.py --h # if you want to train GeoNetM
```

### Use a Pre-trained Model

### Resampling
import resample/resampling.rectification to use the function to resample the distorted image by the forward flow.

The distorted image should be a numpy array with the shape of H\*W\*3 for a color image or H\*W for a greyscale image, the forward flow should be an array with the shape of 2\*H\*W.

The function will return the resulting image and a mask to indicate whether each pixel will converge within the maximum iteration.
## Citation
```bash
@inproceedings{li2019blind,
  title={Blind Geometric Distortion Correction on Images Through Deep Learning},
  author={Li, Xiaoyu and Zhang, Bo and Sander, Pedro V and Liao, Jing},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4855--4864},
  year={2019}
}
```
