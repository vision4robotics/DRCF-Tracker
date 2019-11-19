# Real-Time UAV Objuect Tracking with Saliency-Based Dynamical Dual Regularized Correlation Filter

Matlab implementation of our Saliency-Based Dynamical Dual
Regularized Correlation Filter  (DRCF) tracker.

# Abstract 
Albeit the competitive performance from recent discriminative correlation filters (DCF), their tracking abilities are limited by the corrosive boundary effect inherited from correlation operations. To address this issue, we propose a dual regularized correlation filter tracker with a novel parallel regulation strategy using saliency detection based dynamic regularizers for UAV visual tracking applications, i.e., DRCF tracker. 

Compared to conventional spatial regularized DCFs, we introduce the novel dual regularizer inside the core of the filter generating ridge regression for the DRCF. By using the dual regularizer in the regression, the DRCF can evidently penalize the irrelevant background noises coefficients and enhance the discriminative ability of the filter by suppressing the corrosive boundary effect. On top of the parallel regulation strategy,
we employ a method based on saliency detection to generate the regularizers dynamically and grant the regularizers online adjusting ability. By the merit of the saliency method, the generated dynamic regularizers are able to distinguish the object from the background and to actively regularize the filter according to the changing shape of the object. 

Qualitative and quantitative experiments on multiple challenging UAV tracking benchmarks demonstrate that our DRCF tracker performs favorably against state-of-the-art methods in terms of accuracy and robustness with a real-time speed of 38.4 FPS on a single
CPU.

# Installation

### Using git clone

1. Clone the GIT repository:

   

2. Run the demo script in MATLAB to test the tracker:

   |>> demo_DRCF

   

3. Demo videos is available at : https://www.youtube.com/watch?v=hEUgu_kVSW8

## Description and Instructions

### How to run

The files in root directory are used to run the tracker in UAV123@10FPS, UAVDT and other similar standard datasets.

* run_DRCF.m  -  runfile for the DRCF tracker with hand-crafted features (i.e., HOG+CN).

Tracking performance on the UAV123@10FPS and UAVDT is given as follows:

![Precision Plot on UAV123@10FPS benchmark](https://github.com/sadwfi/DRCF-tracker-upload/raw/master/results/precision%20UAV123%4010FPS.png)  

![Success Plot on UAV123@10FPS benchmark](https://github.com/sadwfi/DRCF-tracker-upload/raw/master/results/success%20UAV123%4010FPS.png)

![Precision Plot on UAVDT benchmark](https://github.com/sadwfi/DRCF-tracker-upload/raw/master/results/precision%20UAVDT.png)

![Success Plot on UAVDT benchmark](https://github.com/sadwfi/DRCF-tracker-upload/raw/master/results/success%20UAVDT.png)

### Features

1. HOG features. It uses the PDollar Toolbox [2], which is included as a git submodule in external_libs/pdollar_toolbox/.

2. Lookup table features. These are implemented as a lookup table that directly maps an RGB or grayscale value to a feature vector.

3. Colorspace features. Currently grayscale and RGB are implemented.

## Acknowledgements

We thank for Dr. martin Danelljan, Dr.Hamed Kiani and  Dr. Feng  Li for their valuable help on our work. In this work,
we have borrowed the feature extraction modules from the ECO tracker (https://github.com/martin-danelljan/ECO) and the parameter settings from BACF (www.hamedkiani.com/bacf.html).

Besides, we have reference the scale estimation module from the DSST tracker (https://github.com/gnebehay/DSST) and the iterative solving procedure programming in STRCF tracker (https://github.com/lifeng9472/STRCF).

## References

[1]  Demo videos is available at : https://www.youtube.com/watch?v=hEUgu_kVSW8