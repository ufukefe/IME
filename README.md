# Image Matching Evaluation (IME)

IME provides to test any feature matching algorithm on datasets containing ground-truth homographies. 

Also, one can reproduce the results given in our paper [*Effect of Parameter Optimization on Classical and Learning-based Image Matching Methods*](https://openaccess.thecvf.com/content/ICCV2021W/TradiCV/papers/Efe_Effect_of_Parameter_Optimization_on_Classical_and_Learning-Based_Image_Matching_ICCVW_2021_paper.pdf) published in [ICCV 2021 TradiCV Workshop.](https://sites.google.com/view/tradicv) 

## Currently Supported Algorithms

| **Classical** | **Learning-Based** |
|:---------:|:--------------:|
| SIFT      | SuperPoint     |
| SURF      | SuperGlue      |
| ORB       | Patch2Pix      |
| KAZE      | DFM            |
| AKAZE     |                |
    
## Environment Setup
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

1. For DFM algorithm you can arrange ratio test threshold using DFM/python/algorithm_wrapper_util.py 
by changing ratio_th (default = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0]). 

    For all classical algorithms you can arrange ratio test threshold by changing the ratio parameter of mnn_ratio_matcher function in algorithm_wrapper_util.py for each algortihm.

    For SuperPoint again you should change ratio parameter of mnn_ratio_matcher function in algorithm_wrapper.py

    For Patch2Pix you should change io_thres parameter in algorithm_wrapper_util.py

2. Use get_names.py to select algorithms and datasets.

3. You can put your own algorithm on Algorithm folder to evaluate with creating a wrapper with the same format. This wrapper should output the matched pixel positions between two images using the selected algorithm.

4. You can put your own dataset on Dataset folder to evaluate by arranging the proper format. Dataset should be in the form of Dataset/subset/subsubset/

## Reproducing Results Given in our Paper

We provide the results given in our paper in ICCV_Results folder. To reproduce the results, you can run an experiment for a specific ratio test or confidence threshold and copy the results in the relevant ratio threshold folder in hpatches_classical or hpatches_deep folder. Then, you can run rt_fig.py and auc_fig.py scripts to save and view the figures.

## TODO
Algorithms to be added:
- [LoFTR](https://zju3dv.github.io/loftr/) ([LoFTR Kornia](https://kornia-tutorials.readthedocs.io/en/latest/image_matching.html))
- [GFTTAffNetHardNet](https://kornia.readthedocs.io/en/latest/_modules/kornia/feature/integrated.html#LocalFeatureMatcher)

Datasets to be added:
- [Multi Modality Dataset](https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods)


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
