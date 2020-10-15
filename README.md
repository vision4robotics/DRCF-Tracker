# Object Saliency-Aware Dual Regularized Correlation Filter for Real-Time Aerial Tracking 

Matlab implementation of our Saliency-Aware Dual Regularized Correlation Filter (DRCF) tracker.

# Publishment and Citation

This paper has been published by IEEE TGRS.

You can find this paper here: https://ieeexplore.ieee.org/document/9094040.

Please cite this paper as: 

@ARTICLE{9094040,

  author={C. {Fu} and J. {Xu} and F. {Lin} and F. {Guo} and T. {Liu} and Z. {Zhang}},
  
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  
  title={Object Saliency-Aware Dual Regularized Correlation Filter for Real-Time Aerial Tracking}, 
  
  year={2020},
  
  volume={},
  
  number={},
  
  pages={1-12}
  
  }
  
# Abstract 
Spatial regularization 1 has been proved as an effective method for alleviating the boundary effect and boosting the performance of a discriminative correlation filter (DCF) in aerial visual object tracking. However, existing spatial regularization methods usually treat the regularizer as a supplementary term apart from the main regression and neglect to regularize the filter involved in the correlation operation. To address the aforementioned issue, this article introduces a novel object saliency-aware dual regularized correlation filter, i.e., DRCF.

Specifically, the proposed DRCF tracker suggests a dual regularization strategy to directly regularize the filter involved with the correlation operation inside the core of the filter generating ridge regression. This allows the DRCF tracker to suppress the boundary effect and consequently enhance the performance of the tracker. Furthermore, an efficient method based on a saliency detection algorithm is employed to generate the dual regularizers dynamically and provide the regularizers with online adjusting ability. This enables the generated dynamic regularizers to automatically discern the object from the background and
20 actively regularize the filter to accentuate the object during its unpredictable appearance changes. By the merits of the dual regularization strategy and the saliency-aware dynamical regularizers, the proposed DRCF tracker performs favorably in terms of suppressing the boundary effect, penalizing the irrelevant background noise coefficients and boosting the overall performance of the tracker. 

Exhaustive evaluations on 193 challenging video sequences from multiple well-known challenging aerial object tracking benchmarks validate the accuracy and robustness of the proposed DRCF tracker against 27 other state-of-the-art methods. Meanwhile, the proposed tracker can perform real-time aerial tracking applications on a single CPU with sufficient speed of 38.4 frames/s.



# Contact

Changhong Fu 

Email: cahnghongfu@tongji.edu.cn

Juntao Xu

Email: ray_xujuntao@tongji.edu.cn

# Installation

Run the demo script to test the tracker:

|>> demo_DRCF

## Description and Instructions

### How to run

The files in root directory are used to run the tracker in UAVDT and UAV123@10fps datasets.

These files are included:

* run_DRCF.m  -  runfile for the DRCF tracker with hand-crafted features (i.e., HOG+CN).

### Features

1. HOG features. It uses the PDollar Toolbox [2], which is included as a git submodule in external_libs/pdollar_toolbox/.

3. Lookup table features. These are implemented as a lookup table that directly maps an RGB or grayscale value to a feature vector.

4. Colorspace features. Currently grayscale and RGB are implemented.

## Acknowledgements

We thank for Dr. `Martin Danelljan` ,`Li Feng` and  `Hamed Kiani` for their valuable help on our work. In this work,
we have borrowed the feature extraction modules from the ECO tracker (https://github.com/martin-danelljan/ECO) and the parameter settings from BACF (www.hamedkiani.com/bacf.html) and STRCF ( https://github.com/lifeng9472/STRCF ) .
