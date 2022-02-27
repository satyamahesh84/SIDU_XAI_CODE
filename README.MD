## SIDU: Similarity Difference And Uniquness Method for Explainable AI

This repository contains source code necessary to reproduce some of the main results in the paper:

If you use this software in an academic article, please cite this paper: <br>

 
**SIDU: Similarity Difference And Uniqueness Method for Explainable AI** (https://doi.org/10.1016/j.patcog.2022.108604)  (https://ieeexplore.ieee.org/abstract/document/9190952) (https://arxiv.org/abs/2006.03122)


This folder contains supporting for code developed in python.


To use this code you need install supporting modules of python.


1. tensorflow gpu

2. numpy

3. scipy

4. tqdm

5. os

6. skimage

7. PIL

8. matplotlib


Further details can be found on the paper as follow,

**If you use this software in an academic article, please consider citing:**

		

           @article{MUDDAMSETTY2022108604,
                    title = {Visual explanation of black-box model: Similarity Difference and Uniqueness (SIDU) method},
                    journal = {Pattern Recognition},
                    volume = {127},
                    pages = {108604},
                    year = {2022},
                    issn = {0031-3203},
                    doi = {https://doi.org/10.1016/j.patcog.2022.108604},
                    url = {https://www.sciencedirect.com/science/article/pii/S0031320322000851},
                    author = {Satya M. Muddamsetty and Mohammad N.S. Jahromi and Andreea E. Ciontos and Laura M. Fenoy and Thomas B. Moeslund},
                    keywords = {Explainable AI (XAI), CNN, Adversarial attack, Eye-tracker}

         }

         @inproceedings{9190952,
		  author={Muddamsetty, Satya M. and Mohammad, N. S. Jahromi and Moeslund, Thomas B.},
		  booktitle={2020 IEEE International Conference on Image Processing (ICIP)}, 
		  title={SIDU: Similarity Difference And Uniqueness Method for Explainable AI}, 
		  year={2020},
		  volume={},
		  number={},
		  pages={3269-3273},
		  doi={10.1109/ICIP40778.2020.9190952}
		}


For more information regarding the paper check out the below diagrams,


## Abstract:
Explainable Artificial Intelligence (XAI) has in recent years become a well-suited framework to generate human understandable explanations of ‘black- box’ models. In this paper, a novel XAI visual explanation algorithm known as the Similarity Difference and Uniqueness (SIDU) method that can effectively localize entire object regions responsible for prediction is presented in full detail. The SIDU algorithm robustness and effectiveness is analyzed through various computational and human subject experiments. In particular, the SIDU algorithm is assessed using three different types of evaluations (Application, Human and Functionally-Grounded) to demonstrate its superior performance. The robustness of SIDU is further studied in the presence of adversarial attack on ’black-box’ models to better understand its performance. Our code is available at: https://github.com/satyamahesh84/SIDU_XAI_CODE.


## SIDU BLOCK DIAGRAM
![](https://github.com/satyamahesh84/SIDU_XAI_CODE/blob/main/SIDU_pics/SIDU_block_diagram.jpg)
## Generating Feature Activations Image masks
![](https://github.com/satyamahesh84/SIDU_XAI_CODE/blob/main/SIDU_pics/SIDU_Generating_Feature_Image_Masks.jpg)

## Visual Explanations 
![](https://github.com/satyamahesh84/SIDU_XAI_CODE/blob/main/SIDU_pics/Visual_Explanation_of%20Natural_Image_SIDU.png)



