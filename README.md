[![Generic badge](https://img.shields.io/badge/Made_With-Python-<COLOR>.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Library-pymc-red.svg)](https://shields.io/)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
![visitor badge](https://visitor-badge.glitch.me/badge?page_id=o-ikne.BML-Project)

# **Bayesian Machine Learning Project**

## **1. Overview**
This [project](https://github.com/rbardenet/bml-course/blob/m2-lille/projects/papers.pdf) is part of the lecture on *Bayesian Machine Learning* tought by [Rémi BARDENET](https://rbardenet.github.io/). The idea is to pick a paper from a list of given papers, and read it with a critical mind. For instance, we will:
- (1) explain the contents of the paper
- (2) emphasize the strong and weak points of the paper
- (3) apply it to real data of our choice when applicable.

This repository contains our Python implementation for this project.
>This project is done by Zakaria Boulkhir \& me. For more insights check out our report.

## **2. Article**
M. Welling and Y. W. Teh. Bayesian learning via stochastic gradient Langevin
dynamics. In Proceedings of the 28th international conference on machine learning
(ICML-11), pages 681–688, 2011. 

```
@inproceedings{welling2011bayesian,
  title={Bayesian learning via stochastic gradient Langevin dynamics},
  author={Welling, Max and Teh, Yee W},
  booktitle={Proceedings of the 28th international conference on machine learning (ICML-11)},
  pages={681--688},
  year={2011},
  organization={Citeseer}
}
```
>[paper](http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf)

## **3. Data**
During this project, we worked with both artificial and real data. For the Mixture of Gaussians we will generate a number of samples we will be working with. For the *Logistic Regression* experiment, we will use the [*a9a*](https://www.openml.org/d/1430) dataset from the UCI *adult* dataset. Finally for the *ICA* experiments, we will use both a synthetic and real data.

## **4. Experiments**

### **Mixture of Gaussians**
To show that our method works well, we start by applying it on a very basic and simple example with two parameters. This first example is the mixture of Gaussians.

### **Logistic Regression**
For this second example, we apply stochastic gradient Langevin algorithm to a Bayesian logistic regression model.
We will be using the data from the *UCI* dataset, more specificaly, the *a9a* dataset which consists of 32561 observations and 123 features.

### **Independent Components Analysis**
The last experiment concerns ICA algorithm based on stochastic (natural) gradient optimization (Amari et al., 1996).

## **5. Installation**

To try our implementation in your own machine you need to install the following requirements:

```python
pip install -r requirements.txt
```
