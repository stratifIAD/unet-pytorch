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

## Data available
|   **Dataset 1**   |   **Dataset 2**   |
|:-----------------:|:-----------------:|
|       6 WSI       |       8 WSI       |
| ALZ50 antibodies  | AT8 antibodies    |
| NanoZoomer 2.0-RS | NanoZoomer 2.0-RS & S60 |
| 5000 manual annotations ||
| 128x128 patch-size | 128x128 & 256x256 patch-size |
| 20x (227 nm/px @ 40x) | 20x (227 & 221 nm/px @ 40x) |
[Properties of the datasets used in the study. For details about the sampling protocols and data augmentation please refer to the article.]

## Results


## Application


## How to cite this work:
This project was published in MICCAI 2022. Use the following citation is you used the github code and/or the dataset associated to it. 

```
@InProceedings{miccai-paper-2116,
author="Jimenez, Gabriel and Kar, Anuradha and Ounissi, Mehdi and Ingrassia, Léa and Boluda, Susana and Delatour, Benoît and Stimmer, Lev and Racoceanu, Lev",
title="Visual DL-based explanation for neuritic plaques segmentation in Alzheimer's Disease",
booktitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer International Publishing",
}
```

