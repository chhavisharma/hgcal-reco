# HGCal Recostruction
ML based Reconstruction for HgCal Data.
A tracker for work done at Ferrmilabs.


### Execution
* Author: [Chhavi Sharma](https://www.linkedin.com/in/chhavi275/)
* Tested on: *TBD*
* Installation Instructions: *TBD*
* Execution Instructions: *TBD*


### Summary 
An attempt to learn algorithms that group detector signatures from the same particle together, and then synthesize those collected data into physically meaningful quantities. Or in simpler terms - An attempt to reconstruct particle Hits to study the properties of LHC particle collision showers in the CMS High-Granularity Calorimeter.

<p align="center"> CMS High-Granularity Calorimeter </p>
<p align="center">
  <img src="img/hgcal.PNG" width="400"/>
</p>   

**HgCal** or High-Granularity Calorimeter is the latest detector used to record particle hits as they flow/disntegrate/shower through the it from the [LHC](https://home.cern/science/accelerators/large-hadron-collider). [Calorimetry](https://cms.cern/news/new-era-calorimetry) - the process of measuring the amount of heat released or absorbed during a chemical reaction - is used to record the particle signature at each hit through the HgCal layers. The amount of energy lost along with timestap and exact spatical coordinates are recorded which detail out the evolution of the showers. 

Little is known about how the HgCal would reconstruct (hypothetical/ new!) non-standard model particles that make weird signatures in the calorimeter, so highlighting the non-confirming particles using multiclass deep neural nets would be an interesting approach.


**Exisitng Pipeline**
Graph Neural Networks are used to model raw HGCal data into particle shower clusters in these 4 categories.
  - 0 Noise
  - 1 Hadrons: ...
  - 2 EM Particles: ...
  - 3 Mips: ...

<p align="center"> Existing Pipeline </p>
<p align="center">
  <img src="img/pipeline.PNG" width="400"/>
</p>  

### Tentative Roadmap

Optimize Existing Pipeline
  - Hyperparameter optimization for Segmentation (EdgeNet) and Pooling (Dynamic Reduction Net)
    - Optimal Neighbours for Graph CNN Accumulation
    - Optimal number of Pooling Layers
    - Mean pool v/s Avg Pool 

Explore Alternate Reconstruction Methods
  - End-to-end gradient flow with the current pipeline 
    - Infuse graph Segregation in the deep network for gradient flow from Reduction all the way to Segmentation
  - Energy Regression
    - Attention/ transformer based single shot GNN
    - Draw from Object Condensation (by Jan K.)
    - ASAP - Supervised clustering without a limit on K



### Weekly Tracking

#### Week 1 

#### Week 2

#### Week 3

#### Week 4


### Resources and References 

1. On the more general side of things:
        The "paper" [https://arxiv.org/abs/1611.08097](https://arxiv.org/abs/1611.08097)
        A good review paper: [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)

2. The kinds of networks we typically use:
        https://arxiv.org/abs/1612.00222 
        https://arxiv.org/abs/1801.07829  
        https://arxiv.org/abs/1902.07987 
        https://arxiv.org/abs/2003.08013  

3. One specialized loss that is curious but fiddly:
        https://arxiv.org/abs/2002.03605 OBJECT CONDENSATION

4. More specific example papers:
        https://arxiv.org/abs/2003.11603  [Graph Neural Networks for Particle Reconstruction in High Energy Physics detectors]
 
5. Some possibly interesting directions to go:
        https://arxiv.org/abs/1801.02144 <— Covariant Compositional Networks For Learning Graphs
        https://arxiv.org/abs/1911.07979 <-- this one I am very curious to try, it's quite similar to object 
        condensation, etc. but better formulated [ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical GraphRepresentations]

6. For deployment stuff, we're heavily into pytorch and nvidia triton:
        https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/ 
        https://github.com/rusty1s/pytorch_geometric/pull/1191 (follow the links further down to see where it actually gets integrated)


