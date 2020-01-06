# Object Saliency-Aware Dual Regularized Correlation Filter for Real-Time Aerial Tracking 

Matlab implementation of our Saliency-Aware Dual Regularized Correlation Filter (DRCF) tracker.

# Abstract 
Spatial regularization has proven itself to be an effective method in terms of alleviating the corrosive boundary effect and boosting the performance of discriminative correlation filter (DCF) in arial visual object tracking. 

However, existing spatial regularization methods usually treat the regularizer as a supplementary term aside to the main regression and neglect to regularize the filter involved in the correlation operation. 

To mainly address the aforementioned issue, this work introduces an novel object saliency-aware dual regularized correlation, i.e., DRCF. 

Specifically, the proposed DRCF tracker suggests a dual regularization strategy to directly regularize the filter involved with the correlation operation inside the core of the filter generating ridge regression. 

This allows the DRCF tracker to evidently suppress the boundary effect and consequently enhance the performance of the tracker. 

Furthermore, an efficient method based on saliency detection algorithm is employed to generate
the dual regularizers dynamically and provide the regularizers with online adjusting ability. 

This enables the generated dynamic regularizers to automatically discern the object from the background
and actively regularize the filter to accentuate the object during its unpredictable appearance changes. 

By the merits of the dual regularization strategy and the saliency-aware dynamical regularizers, the proposed DRCF tracker performs favourably in terms of suppressing the pernicious boundary effect, penalizing the irrelevant background noises coefficients and boosting the overall performance of the tracker. 

Exhaustive evaluations on 173 challenging video sequences from multiple well-known aerial object tracking benchmarks validate the accuracy and robustness of the proposed DRCF tracker against other state-of-the-art methods. 

Meanwhile, the proposed tracker can perform real-time aerial tracking applications on single CPU with sufficient speed of 38.4 frames per second.



# Contact

Changhong Fu 

Email: cahnghongfu@tongji.edu.cn

Juntao Xu

Email: ray_xujuntao@tongji.edu.cn

# Installation

1. Clone this git repository: https://github.com/vision4robotics/DRCF-Tracker.git
2. Start Matlab and navigate to the repository of DRCF_v2 for the latest version of the tracker.
3. Run the demo script to test the tracker:

|>> demo_DRCF

## Description and Instructions

### How to run

The files in DRCF_v2 directory are used to run the tracker in UAVDT and UAV123@10fps datasets.

These files are included:

* run_DRCF.m  -  runfile for the DRCF tracker with hand-crafted features (i.e., HOG+CN).

  

Tracking performance on the UAVDT and UAV123@10fps is given as follows：

Note that the following results are obtained using the latest DRCF.V2 version of the tracker. 

 

![Image text](https://raw.githubusercontent.com/sadwfi/DRCF_2020/master/OverallBenchmarkResults/error_OPE_UAVDT.png)



![Image text]( https://raw.githubusercontent.com/sadwfi/DRCF_2020/master/OverallBenchmarkResults/overlap_OPE_UAVDT.png) 

![Image text](  https://raw.githubusercontent.com/sadwfi/DRCF_2020/master/OverallBenchmarkResults/error_OPE_UAV123.png ) ![Image text]( https://raw.githubusercontent.com/sadwfi/DRCF_2020/master/OverallBenchmarkResults/overlap_OPE_uav123.png)

### Features

1. HOG features. It uses the PDollar Toolbox [2], which is included as a git submodule in external_libs/pdollar_toolbox/.

3. Lookup table features. These are implemented as a lookup table that directly maps an RGB or grayscale value to a feature vector.

4. Colorspace features. Currently grayscale and RGB are implemented.

## Acknowledgements

We thank for Dr. `Martin Danelljan` ,`Li Feng` and  `Hamed Kiani` for their valuable help on our work. In this work,
we have borrowed the feature extraction modules from the ECO tracker (https://github.com/martin-danelljan/ECO) and the parameter settings from BACF (www.hamedkiani.com/bacf.html) and STRCF ( https://github.com/lifeng9472/STRCF ) .