---
layout: archive
title: "StratifIAd"
permalink: /projects/stratifiad-miccai2022
author_profile: true
---

## Visual deep learning-based explanation for neuritic plaques segmentation in Alzheimer’s Disease using weakly annotated whole slide histopathological images  
Gabriel Jimenez, Anuradha Kar, Mehdi Ounissi, Léa Ingrassia, Susana Boluda, Benoît Delatour, Lev Stimmer, and Daniel Racoceanu

## About the project
- The problem:
    - AD is caused by misfolding and accumulation of Amyloid-beta peptides and tau proteins.
    - The clinical presentation of the patients is very heterogeneous.
    - In vivo imaging methodologies do not offer the resolution obtained by microscopy.

- Findings:
    - Tau pathologies (tauopathies) are highly linked to clinical manifestations of AD [Duyckaerts, C., et al. Classification and basic pathology of Alzheimer disease. Acta Neuropathologica, 2009.]

- The idea:
    - Automatic segmentation of tau protein aggregates (neuritic plaques) from WSI.
    - Manual annotation refinement using deep learning attention mechanisms and visual explainability features from DL models.
    - Evaluate the impact of context information and antibodies on the DL performance.

## Dataset
Properties of the datasets used in the study. For details about the sampling protocols and data augmentation please refer to the [article](https://export.arxiv.org/abs/2302.08511).

|   **Dataset 1**   |   **Dataset 2**   |
|:-----------------:|:-----------------:|
|       6 WSI       |       8 WSI       |
| ALZ50 antibodies  | AT8 antibodies    |
| NanoZoomer 2.0-RS | NanoZoomer 2.0-RS & S60 |
| 128x128 patch-size | 128x128 & 256x256 patch-size |
| 20x (227 nm/px @ 40x) | 20x (227 & 221 nm/px @ 40x) |

## Results

See all the results and analysis in our [github](https://github.com/aramis-lab/miccai2022-stratifiad) and [article](https://export.arxiv.org/abs/2302.08511).

<p align="center">
    <img width="80%" src="https://github.com/stratifIAD/unet-pytorch/blob/098f742f58799e4ff277d0f4f1c6ecb32e257ed1/imgs/patchsize.png">
    <br>Example of plaque image for different levels of context. It was shown that context information impacted on the network perfomance.
</p>

<p align="center">
    <img width="80%" src="https://github.com/stratifIAD/unet-pytorch/blob/098f742f58799e4ff277d0f4f1c6ecb32e257ed1/imgs/att_unet.png">
    <br>Focus progression using successive activation layers of attention-UNet. This model proved to be useful for improving the manual annotations of neuritic plaques.
</p>

## Application

<p align="center">
    <img width="80%" src="https://github.com/stratifIAD/unet-pytorch/blob/098f742f58799e4ff277d0f4f1c6ecb32e257ed1/imgs/stratifiad-system.png">
    <br>Final application using the deep learning pipelines developed in the project. The main goal is to achieve patient stratification.
</p>

## Key ideas from the study
1. Antibodies can impact the detection and segmentation of tau aggregates in WSI.
    - AT8 creates less compact structures making segmentation of plaques challenging. 

2. WSI are frequently acquired using different scanners having different properties.
    - Amplification of human-software annotation errors.

3. The context effect in segmentation
    - Related to the negative impact of larger patch size.

4. Visual explainability to improve manual annotations.
    - Improve morphological analysis of tau aggregates.

5. Benchmark with commercial software
    - Outperformed in test dataset. Commercial software follows a black-box approach; therefore, no explainability

## How to cite this work:

This project was published in MICCAI 2022. Use the following citation if you used the [github](https://github.com/aramis-lab/miccai2022-stratifiad) code and/or the dataset associated to it. 

```
@InProceedings{miccai-paper-2116,
author="Jimenez, Gabriel and Kar, Anuradha and Ounissi, Mehdi and Ingrassia, Léa and Boluda, Susana and Delatour, Benoît and Stimmer, Lev and Racoceanu, Lev",
title="Visual DL-based explanation for neuritic plaques segmentation in Alzheimer's Disease",
booktitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer International Publishing",
}
```

