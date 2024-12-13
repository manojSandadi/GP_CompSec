<<<<<<< HEAD
Codebase for the paper:  

[**On the Robustness of Large Multimodal Models Against Image Adversarial Attacks**](https://arxiv.org/pdf/2312.03777)  [CVPR 2024]

Authors: Xuanming Cui, Alejandro Aparcedo, Young Kyun Jang, Ser-Nam Lim  

### Requirements

```bash
torch, torchvision, transformers==4.31.0, tqdm, accelerate, numpy<2.0.0
```

First install requirements with 

```bash
pip install -r requirements.txt
```

Then build the LAVIS requirements by

```bash
cd models/LAVIS
pip install -e .
```


### Code

- ```attacks/```:the attacks used in the paper

- ```data/```: contains dataset annotations (.jsonl) in the format of {"image": <path/to/image_folder>, "text": <class_name or caption>}. For VQA datasets: mme, pope, sqa, textvqa and vqav2, their corresponding folders also contain scripts to generate and evaluate answers.

- ```models/```: the LMMs used in the project

- ```scripts/```: scripts to generate/evaluate original/adversarial images. under scripts/slurm we also include slurm scripts to run the corresponding python codes. You will have to change the necessary configurations (e.g. conda env, slurm configs).

By default we save the images under ```datasets/```, generated adversarial images under ```adv_datasets/<dataset_name>/<task>/<attack>/<run_name>/```, lmm-generated answers under ```results/responses``` and logs under ```results/logs/<dataset>/<task>/<attack>/<run_name>/```.

### Run

1. Data:
   - COCO: we use COCO2014 validation set. We provide the caption file under data/coco/coco_2014val_caption.json
   - Imagenet: we use Imagenet 2012 validation set and randomly select 5000 samples for testing.
   - Food101: we use its test set.
   - Stanford Cars: we use its train set because the test set does not contain ground-truth.
   - VQA: we use vqav2, SQA, POPE, TextVQA and MME. The download and setup can be found under the original LLaVA [codebase](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
3. Attacks: we use attacks implemented from [torchattack](https://github.com/Harry24k/adversarial-attacks-pytorch). We use two parameter settings: strong and normal. Default parameters can be found in attacks/config.yaml.
4. Generate adversarial images
   - To generate adversarial with different tasks (classification/retrieval/classification_with_context), run
     ```bash
     python scripts/generate_or_evaluate_adversarials.py \
            --model-path ${model_path} \
            --dataset "$dataset" \
            --model-type pretrain \
            --save_image \
            --image_ext 'jpg' \
            --task $task \
            --attack_name $attack_name \
            --attack_params $attack_params \
            --batch_size $batch_size \
            --num_workers $num_workers \
     ```
   - The ```model_path``` is the huggingface-like model signature, e.g. ```openai/clip-vit-large-patch14```,  ```blip2_feature_extractor```, ```liuhaotian/llava-v1.5-13b```, or ```blip2_vicuna_instruct```. The ```model-type``` is for selecting the LLM for BLIP models.
   - The image_ext is the format of the image file, as normal image extension (.jpg, .png etc.) or .pt files (adversarial imaes).
   - The ```attack_name``` has to be one defined and exported in attacks/__init__.py. The ```attack_param``` can be one of ['normal', 'strong'], or a string of dictionary of parameters that will be passed to the target attack method.

6. Evaluate
   - To evaluate for classification, run the above code but do not pass attack_name.
   - To evaluate caption retrieval, use ```scripts/evaluate_caption_retreieval.py```. Checkout ```scripts/eval_caption_retrieval_llava(blip).sh``` for sample runs.
   - To evaluate VQA, please refer to each dataset under ```data/```.

### Acknowledgement

This repository is using code from [LLaVA](https://github.com/haotian-liu/LLaVA), [LAVIS](https://github.com/salesforce/LAVIS),  [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).
=======
# Analyzing the Impact of Adversarial Attacks on Multimodal Models: A Study of Targeted and Untargeted Approaches

# Implementation Details:
In this project, we will showcase how adversarial attacks affect the LLaVA model, a multimodal model that processes both images and text. Our focus will be on using both targeted and untargeted adversarial attack methods to evaluate the model's performance under different conditions. We will start by implementing well-known techniques such as the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) to create adversarial examples that challenge the model.

To conduct our research, we will prepare a dataset containing a variety of image-text pairs, which are typical inputs for the LLaVA model. This dataset will enable us to examine how the model responds when faced with adversarial examples. We will test the model using both targeted and untargeted attacks, measuring its accuracy, precision, and recall before and after introducing the adversarial examples. This approach will provide insights into how well the model can classify and interpret inputs in the presence of adversarial challenges.

In analyzing the effects of these attacks on the LLaVA model, we will focus on illustrating the overall impact of adversarial techniques on its performance rather than identifying specific vulnerabilities. We will create visualizations that demonstrate how the model’s outputs change when subjected to different attack scenarios. This will include comparing the model's responses to standard inputs versus adversarial inputs, highlighting any significant differences in classification results. By documenting these changes, we aim to provide a clearer understanding of the challenges multimodal models face in real-world applications. Ultimately, our goal is to contribute valuable insights into the implications of adversarial attacks for the security and reliability of machine learning models, fostering a better understanding of how these models can be tested and improved against potential threats.
>>>>>>> 34fc7fe4bcd6878ea05e5fb752a93c05f920c90a
