# Reproducibility study for CISC867 Fall 2021

This repository is the reproducibility study of [Deep Just-In-Time Inconsistency Detection Between Comments and Source Code](https://arxiv.org/pdf/2010.01625.pdf). 


## Requirements

None! Our code is currently hosted on Google Colab, and all libraries and dependencies will be installed through the Colab runtime.


## Training

Training of the model is also done on the Colab environment. The trained model can be viewed by following the Colab links.


## Evaluation

The hyperparameters used were in accordance with those described by Panthaplackel et al. in their paper, also summarized below:

| Layer				| Parameters				| Value		|
| ----------------- | ------------------------- | --------- |
| Embedding			| Dimension:				| 64		|
| BiGRU	Encoder		| Hidden dimension: 		| 64		|
| GGNN	Encoder		| Hidden dimension: 		| 64		|
| GGNN	Encoder		| Message-passing steps: 	| 8			|
| Multi-Attention	| Attention heads:			| 4			|
| BiGRU Decoder		| Hidden dimension:			| 64		|
| Dense				| Dropout:					| 0.6		|


## Results

The baselines implementation can be found within this repository.

The implementation and results for the models without features can be found at: https://colab.research.google.com/drive/11Vmawr0xTka7-DdSpBF-5Tf9FGdp3JYw?usp=sharing

The implementation and results for models with features can be found at: https://drive.google.com/file/d/1pDt6F-7iTYAVO9wYenLD18jaVFJldwMp/view?usp=sharing