## HGCal Recostruction

ML based Reconstruction for HCCAL Data. 
Here we are using the TrackML dataset as a testing ground for a panoptic segmentation task, which is assigning "hits" from a particle physics detector into "tracks" or otherwise clusters and also assigning various properties to the hits and found clusters.
Once there is suitable ground truth for HGCAL we'll move to that.
This work is supported by LDRD grant 2019-017 funded by Fermilab.

Current:\
Experiments with end-to-end reconstruction/segmentation/regression model design, with a simpler dataset for proof of concept. \
Dataset: [kaggle page](https://www.kaggle.com/c/trackml-particle-identification/overview)\
Experiments: [one_shot_tests](one_shot_tests/embedding_with_graphconv.py)

### Installation

To get all necessary software please run `install.sh`!

You will also need to have a kaggle account and setup the kaggle api to get the dataset we're using!
You can find detailed instruction for setting up the kaggle API tokens [here](https://adityashrm21.github.io/Setting-Up-Kaggle/).

The TrackML dataset is very large on disk. You will need at least 300 GB of space to hold the full dataset locally.
The default dataset that is retrieved is a ~1GB sub-sample of the training data which is enough to confirm the software works, but not really train a functioning model.

To run the one-shot segmentation model go into `one_shot_tests` and run `python embedding_with_graphconv.py` to start training.


### Index
- [Summary](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#summary)
- [Tentative Roadmap](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#tentative-roadmap)
- [Resources and References](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#resources-and-references)

<!---
### Execution
* Author: [Chhavi Sharma](https://www.linkedin.com/in/chhavi275/)
* Tested on: *TBD*
* Installation Instructions: *TBD*
* Execution Instructions: *TBD*
-->

### Summary 
An attempt to learn algorithms that group detector signatures from the same particle together, and, synthesize it into physically meaningful quantities  facilitating the study of the properties of LHC particle collisions in the [CMS](https://home.cern/science/experiments/cms) High-Granularity Calorimeter.

The [High Granularity Calorimeter](https://cms.cern/news/new-era-calorimetry) is used to record particle hits as they flow/ disintegrate/ shower through the it from the LHC. [Calorimetry](https://cms.cern/news/new-era-calorimetry) - the process of measuring the amount of heat released or absorbed during a chemical reaction - is used to record a particle's signature at each hit through the HgCal layers. The amount of energy lost along with timestap and exact spatial coordinates are recorded which detail out the evolution of the showers. 

An attemp needs to be made to reduce the computation and memory requirements of the reconstruction algorithm along with imporving accuracy and discrimination capability. Little is known about how we would reconstruct (hypothetical/ new!) non-standard model particles that make weird signatures in the calorimeter, so highlighting the non-confirming particles/hits as a seperate class in the learning algorithm would also be an interesting capability.

Ground-truth-labelling: An event consists of a collection of hits over a 25 ns window. Each event is considered as a data sample unit and could have upto 20k particle hits.\
EdgeNet: ...\
Union-find Segregation: ...\
Dynamic Reduction Net: ... 

_________________________________________________________________________________________________


### Tentative Roadmap

- Complete retraining of the pipeline
    - Setup Google Cloud VM
        - Transition to Pytorch 1.5.1 and CUDA 10.2 and corresponding pytorch geometric verison
    - Understand existing pipleine
        - Breakdown data preprocessing and compute statistics
        - Retrain the entire pipline - EdgeNet and Dynamic Reduction Net
    
- Optimize Existing Pipeline
    - Hyperparameter optimization for Segmentation (EdgeNet) and Pooling (Dynamic Reduction Net)
        - Optimal Neighbours for Graph CNN Accumulation
        - Optimal number of Pooling Layers
        - Mean pool v/s Avg Pool 

- Explore Alternate Reconstruction Methods
    - End-to-end gradient flow with the current pipeline 
        - Infuse graph Segregation in the deep network for gradient flow from Reduction all the way to Segmentation
    - Energy Regression
        - Attention/ transformer based single shot GNN
        - Draw from Object Condensation (by Jan K.)
        - ASAP - Supervised clustering without a limit on K

### Resources and References 
0. The standard Model https://home.cern/science/physics/standard-model 
1. GNN Papers: 
    - The "paper" [https://arxiv.org/abs/1611.08097](https://arxiv.org/abs/1611.08097)
    - Review paper: [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)
    - Networks typically used:
      - https://arxiv.org/abs/1612.00222 
      - https://arxiv.org/abs/1801.07829  
      - https://arxiv.org/abs/1902.07987 
      - https://arxiv.org/abs/2003.08013  
    - An interesting specialized loss - OBJECT CONDENSATION https://arxiv.org/abs/2002.03605 
    - More specific example papers- Graph Neural Networks for Particle Reconstruction in High Energy Physics detectors https://arxiv.org/abs/2003.11603 
    - Some interesting directions to go:
      - Covariant Compositional Networks For Learning Graphs https://arxiv.org/abs/1801.02144      
      - ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical GraphRepresentations https://arxiv.org/abs/1911.07979 
    - Juniper (Dianna's Suggestion) Binary Junipr achieves state-of-the-art performancefor quark/gluon discrimination and top-tagging. https://arxiv.org/pdf/1906.10137.pdf
2. Deployment:
    - https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/ 
    - https://github.com/rusty1s/pytorch_geometric/pull/1191
