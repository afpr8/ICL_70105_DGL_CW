# DGL2026 Brain Graph Super-Resolution Challenge

## Contributors

- Adrian Pittaway
- Alejandro Cañada 
- Arunima Singh
- Christian Wood

## Problem Description

Brain graph super-resolution aims to reconstruct high-resolution (HR) connectivity matrices ($\mathbb{R}^{268 \times 268}$) from low-resolution (LR) counterparts ($\mathbb{R}^{160 \times 160}$). This is an essential research area for several reasons:
- Overcoming Data Limitations: High-resolution neuroimaging is costly and noise-prone. Generative models offer a pathway to infer these detailed patterns from more accessible LR data.
- Preserving Topology: The brain functions as a complex, modular network; successful super-resolution must maintain critical topological features like global efficiency and clustering coefficients to ensure the HR graph remains biologically plausible.
- Inductive Learning: Unlike transductive approaches, using an inductive GNN allows for the discovery of generalizable connectivity rules, enabling the model to accurately predict HR graphs for unseen subjects.

## NeuroSRGAN - Methodology

- Summarize in a few sentences the building blocks of your generative GNN model.

- Figure of your model.

## Used External Libraries

- If running the code on a GPU please run pip install -r gpu_requirements.txt
- If running the code on a CPU please run pip install -r cpu_requirements.txt

## Results

- Insert your bar plots.

## References

- [AGSRNet](https://www.researchgate.net/publication/351298271_Brain_Graph_Super-Resolution_Using_Adversarial_Graph_Neural_Network_with_Application_to_Functional_Brain_Connectivity)
