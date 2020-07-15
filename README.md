## HGCal Recostruction

ML based Reconstruction for HgCal Data. A tracker for work done at Ferrmilabs.\
[Presentation](https://docs.google.com/presentation/d/1WW9HBwumZRxsq518BsyRxTFi64VcT6rng09rtjOeNRk/edit?usp=sharing)

### Index
- [Summary](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#summary)
- [Tentative Roadmap](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#tentative-roadmap)
- [Week Jun 29 - Jul 3](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#week-jun-29---jul-3)
- [Week Jun 6  - Jul 10](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#week-jul-6---jul-10)
- [Week Jun 13 - Jul 17](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#week-jun-13---jul-17)
- [Resources and References](https://github.com/chhavisharma/hgcal-reco/blob/master/README.md#resources-and-references)


<!---
### Execution
* Author: [Chhavi Sharma](https://www.linkedin.com/in/chhavi275/)
* Tested on: *TBD*
* Installation Instructions: *TBD*
* Execution Instructions: *TBD*
-->

### Summary 
An attempt to learn algorithms that group detector signatures from the same particle together, and then synthesize those collected data into physically meaningful quantities. Or in simpler terms - An attempt to reconstruct particle Hits to study the properties of LHC particle collision showers in the CMS High-Granularity Calorimeter.

HgCal or High-Granularity Calorimeter is the latest detector used to record particle hits as they flow/disntegrate/shower through the it from the [LHC](https://home.cern/science/accelerators/large-hadron-collider). [Calorimetry](https://cms.cern/news/new-era-calorimetry) - the process of measuring the amount of heat released or absorbed during a chemical reaction - is used to record the particle signature at each hit through the HgCal layers. The amount of energy lost along with timestap and exact spatical coordinates are recorded which detail out the evolution of the showers. 

Little is known about how the HgCal would reconstruct (hypothetical/ new!) non-standard model particles that make weird signatures in the calorimeter, so highlighting the non-confirming particles using multiclass deep neural nets would be an interesting approach.


_________________________________________________________________________________________________


### Tentative Roadmap

- Retrain EdgeNet and Dynamic Reduction Net 
    - Setup Google Cloud VM
    - Transition to Pytorch 1.5.1 and CUDA 10.2
    - Understand existing pipleine
    - Breakdown data preprocessing and compute statistics
    
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

_________________________________________________________________________________________________

### Weekly Tracking


##### Week Jun 29 - Jul 3
 
Exisitng Pipeline: Graph Neural Networks are used to model raw HGCal data into particle shower clusters in these 4 categories.
  - 0 Noise
  - 1 Hadrons: ...
  - 2 EM Particles: ...
  - 3 Mips: ...

<p align="center"> Existing Pipeline </p>
<p align="center">
  <img src="images/ExistingPipeline.png"/>
</p>  

_________________________________________________________________________________________________

##### Week Jul 6 - Jul 10




_________________________________________________________________________________________________

##### Week Jul 13 - Jul 17





_________________________________________________________________________________________________

##### Week Jul 20 - Jul 24





_________________________________________________________________________________________________


### Resources and References 

1. GNN Papers: 
    - The "paper" [https://arxiv.org/abs/1611.08097](https://arxiv.org/abs/1611.08097)
    - A good review paper: [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)
    - Networks typically used:
      - https://arxiv.org/abs/1612.00222 
      - https://arxiv.org/abs/1801.07829  
      - https://arxiv.org/abs/1902.07987 
      - https://arxiv.org/abs/2003.08013  
    - One specialized loss that is curious but fiddly:- https://arxiv.org/abs/2002.03605 OBJECT CONDENSATION LOSS
    - More specific example papers- https://arxiv.org/abs/2003.11603 Graph Neural Networks for Particle Reconstruction in High Energy Physics detectors.
    - Some possibly interesting directions to go:
      - https://arxiv.org/abs/1801.02144 Covariant Compositional Networks For Learning Graphs
      - https://arxiv.org/abs/1911.07979 ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical GraphRepresentations
2. Deployment:
    - https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/ 
    - https://github.com/rusty1s/pytorch_geometric/pull/1191
