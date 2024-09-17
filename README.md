# Similarity of Neural Architectures using Adversarial Attack Transferability (SAT) (ECCV 2024)

Official repository of SAT | [arxiv](https://arxiv.org/abs/2210.11407)

## Introduction

In recent years, many deep neural architectures have been developed for image classification. Whether they are similar or dissimilar and what factors contribute to their (dis)similarities remains curious. To address this question, we aim to design a quantitative and scalable similarity measure between neural architectures. We propose Similarity by Attack Transferability (SAT) from the observation that adversarial attack transferability contains information related to input gradients and decision boundaries widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our SAT to answer the question. In addition, we provide interesting insights into ML applications using multiple models, such as model ensemble and knowledge distillation. Our results show that using diverse neural architectures with distinct components can benefit such scenarios.

## SAT (Similarity by Attack Transferability)

We use the [PGD attack](https://arxiv.org/pdf/1706.06083) to measure attack transferability between two different neural architectures
(here, we denote them as model A and model B).

$$\text{SAT}(A,B) = \log \left( \max \left( \varepsilon_s, 100 \times
\frac{1}{2|X_{AB}|}
\sum_{x \in X_{AB}} \left( \mathbb {I} ({A(x_B)} \neq y) + \mathbb {I} ({B(x_A)} \neq y) \right) \right) \right],$$

where $X_{AB}$ represents the set of inputs that both A and B classify correctly, $x_A$ and $x_B$ are adversarial samples for A and B with 
input $x$, and $\varepsilon_s$ is a small scalar value.

## SAT of 69 Models

Using SAT, we could gain similarity scores between two models among 69 models.
You can find the specific scores [here].

## Architectural Components of 69 Models

To explore what architectural components cause architectural differences, we investigate 13 sub-components of 69 ImageNet classification models.
You can find the list of models and their features [here](https://github.com/J-H-Hwang/SAT/blob/main/Model%20Feature_release.csv).
Note that we follow the corresponding paper and [timm library](https://github.com/huggingface/pytorch-image-models) to list model features.


