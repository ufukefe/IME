# Image Matching Evaluation (IME)
Image Matching Evaluation codes for our [ICCV 2021 TradiCV Workshop](https://sites.google.com/view/tradicv) paper: [*Effect of Parameter Optimization on Classical and Learning-based Image Matching Methods*](https://arxiv.org/pdf/2108.08179.pdf) 

Using the IME, you can test any feature matching algorithm on datasets containing ground-truth homographies. For this, you should create a wrapper for the algorithm to be tested. This wrapper should output the matched pixel positions between the two images using the selected algorithm.

## Currently Supported Algorithms

**Classical** <br />
SIFT <br />
SURF <br />
ORB <br />
KAZE <br />
AKAZE <br />

**Learning-based** <br />
SuperPoint <br />
SuperGlue <br />
Patch2Pix <br />
DFM <br />

## Setup Environment
This repository is created using Anaconda.

Open a terminal in the IME folder and run the following commands;

1. Run bash script to create environment for IME, download algorithms and datasets
````
bash install.sh
````

2. Activate the environment
````
conda activate ime
````

3. Run IME!
````
python3 main.ipy
````
Well done, you can find results on Results folder :)


## Notes: 

1. You can arrange ratio_test_threshold of DFM using DFM/python/algorithm_wrapper_util.py 
by changing ratio_th (default = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0])

2. Use get_names.py to select algorithms and datasets.

3. You can put your own algorithm on Algorithm folder to evaluate with writing a wrapper with the same format.

4. You can put your own dataset on Dataset folder to evaluate by arranging the proper format. Dataset should be in the form of Dataset/subset/subsubset/image1.png













## BibTeX Citation
Please cite our paper if you use the code:

```
@InProceedings{Efe_2021_ICCV,
    author    = {Efe, Ufuk and Ince, Kutalmis Gokalp and Alatan, Aydin},
    title     = {Effect of Parameter Optimization on Classical and Learning-based Image Matching Methods},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
}
```
