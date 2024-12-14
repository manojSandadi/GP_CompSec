# Analyzing the Impact of Adversarial Attacks on Multimodal Models: A Study of Targeted and Untargeted Approaches

# Implementation Details:
In this project, we will showcase how adversarial attacks affect the LLaVA model, a multimodal model that processes both images and text. Our focus will be on using both targeted and untargeted adversarial attack methods to evaluate the model's performance under different conditions. We will start by implementing well-known techniques such as the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) to create adversarial examples that challenge the model.
cd 
To conduct our research, we will prepare a dataset containing a variety of image-text pairs, which are typical inputs for the LLaVA model. This dataset will enable us to examine how the model responds when faced with adversarial examples. We will test the model using both targeted and untargeted attacks, measuring its accuracy, precision, and recall before and after introducing the adversarial examples. This approach will provide insights into how well the model can classify and interpret inputs in the presence of adversarial challenges.

In analyzing the effects of these attacks on the LLaVA model, we will focus on illustrating the overall impact of adversarial techniques on its performance rather than identifying specific vulnerabilities. We will create visualizations that demonstrate how the model’s outputs change when subjected to different attack scenarios. This will include comparing the model's responses to standard inputs versus adversarial inputs, highlighting any significant differences in classification results. By documenting these changes, we aim to provide a clearer understanding of the challenges multimodal models face in real-world applications. Ultimately, our goal is to contribute valuable insights into the implications of adversarial attacks for the security and reliability of machine learning models, fostering a better understanding of how these models can be tested and improved against potential threats.

Code Implementation can be found in visual_adversarial_lmm.
### Acknowledgement
The Code base is made based upon the existing paper [**On the Robustness of Large Multimodal Models Against Image Adversarial Attacks**](https://arxiv.org/pdf/2312.03777)  [CVPR 2024] and their codebase
