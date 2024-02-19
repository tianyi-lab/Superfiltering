# Superfiltering

[Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning](https://arxiv.org/abs/2402.00530)

This is the repo for the Superfiltering project, which introduces a method **astonishingly utilizes small GPT2 model to successfully filter out the high-quality subset from existing GPT4-generated instruction tuning dataset.**

The repo contains:

- The code for Superfiltering.
- The data selected by Superfiltering.
- The model checkpoints (7B) that were trained using our Superfiltering.

(Feel free to email minglii@umd.edu for any questions or feedback.)

## News
- [2024/02] We updated the repo of Superfiltering, code and data are added. 
- [2024/01] We released the Superfiltering paper!

## Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Install](#install)
- [Run Code](#run-code)
- [Data](#data)
- [Evaluation](#evaluation)
- [ToDo](#todo)
- [Citation](#citation)

## Overview

Instruction tuning is critical to improve LLMs but usually suffers from low-quality and redundant data. 
Data filtering for instruction tuning has proved important in improving both the efficiency and performance of the tuning process. 
But it also leads to extra cost and computation due to the involvement of LLMs in this process. 
To reduce the filtering cost, we study Superfiltering: Can we use a smaller and weaker model to select data for finetuning a larger and stronger model?
Despite the performance gap between weak and strong language models, we find their highly consistent capability to perceive instruction difficulty and data selection results. 
This enables us to use a much smaller and more efficient model to filter the instruction data used to train a larger language model. Not only does it largely speed up the data filtering, but the filtered-data-finetuned LLM achieves even better performance on standard benchmarks. 
Extensive experiments validate the efficacy and efficiency of our approach. 

## Highlights

* We reveal the strong consistency between small and large LLMs in perceiving and evaluating the difficulty of instruction tuning data, which provides insights on understanding the differece between small and large mdoels. 
* We propose the first method of Superfiltering that utilizes a small LM, e.g., GPT-2 (124M), to select data for instruction tuning, and brings significant speedups to LLM finetuning pipeline. 
* Superfiltering is plug-and-play method which precises in allocating high-quality and informative data improving LLM instruction tuning. 

## Install

Install the dependencies with `pip install -r requirements.txt`

Note: The calculation of IFD scores only needs the ```transformers``` package, thus if you are using a different code base with ```transformers``` installed, you can directly run the code and manully install the missing packages. 

## Run Code

1. Calculate IFD scores

```
bash scripts/step1_select_data_analysis_gpt2.sh
```

```--data_path```: The targeted dataset in the Alpaca format. <br>
```--save_path```: The path to save the ```.jsonl``` file containing scores. <br>
```--model_name_or_path```: The model used for calculating IFD scores, we found ```gpt2``` is good enough as illustrated in our paper. Also, you can uses the model that you need to finetune, which would be a self-guided manner, or student-envolved manner. 

2. Put scores into original data
```
bash scripts/step2_put_analysis_to_data.sh
```

```pt_data_path```: The ```.jsonl``` file generated in last step. <br>
```json_data_path```: The targeted dataset in the Alpaca format. <br>
```json_save_path```: The data path to save the data with IFD scores. <br>

Note: Step 1 and 2 can be merged directly for better convenience. 

3. Select the data you wish. 
```
bash scripts/step3_select_data.sh
```

```json_data_path```: The data path to save the data with IFD scores. <br>
```json_save_path```: The data path to save the data with IFD scores filtered. <br>
```sample_rate```: How many data you need. Here we only provide the percentage version, you can slghtly modify the code to select exact number you want. 

Note: The Step 1 code is the ```batch_size=1``` version, it takes about 15 minutes to process the whole Alpaca dataset. We release this version and split the whole process into 3 steps for better controllablity. 
You can directly run the above 3 scripts to get a better understand of our codes. 
It takes about 15 minutes for the whole process. 

## Data

The Alpaca Data with GPT2-based IFD scores can be found in ```data/data_with_ifd/alpaca_data_gpt2_data.json```.<br>
The Alpaca-GPT4 Data with GPT2-based IFD scores can be found in ```data/data_with_ifd/alpaca_gpt4_data_gpt2_data.json```.<br>

To select the subset data from these datasets, you can directly run ```bash scripts/step3_select_data.sh``` in above Step 3. 

## Evaluation

The codes and data for pair-wise conparison by using GPT4 is released in the ```evaluation``` folder. 
This method greatly eliminates the potential position bias of GPT4 and chatGPT. 

To use this code, please follow the below scripts:

```bash evaluation/scripts/do_eval_generation.sh```: The model automatically generates the responses for a given instruction in test datasets. <br>
```bash evaluation/scripts/do_eval_generation_wrap.sh```: Wrap the response files of LLMs being compared. <br>
```bash evaluation/scripts/do_eval.sh```: Use GPT4 or chatGPT for the evaluation. <br>
```bash evaluation/scripts/do_review_eval_score.sh```: Parse the results and draw the figure. <be>

For other evaluation metrics, please see their official repo.

## ToDo
- [x] Release the code, data, and models. 
- [ ] Release new versions.
- [ ] Implement our method on more dataset and base models.  

## Citation

Please consider citing our papers if you think our codes, data, or models are useful. Thank you! <br>
The first paper is the Superfiltering paper, and the second one is the Cherry LLM paper, proposing IFD score, which serves as the backbone metric of Superfiltering.

```
@article{Li2024SuperfilteringWD,
  title={Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning},
  author={Ming Li and Yong Zhang and Shwai He and Zhitao Li and Hongyu Zhao and Jianzong Wang and Ning Cheng and Tianyi Zhou},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.00530},
  url={https://api.semanticscholar.org/CorpusID:267365346}
}
```

```
@article{Li2023FromQT,
  title={From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning},
  author={Ming Li and Yong Zhang and Zhitao Li and Jiuhai Chen and Lichang Chen and Ning Cheng and Jianzong Wang and Tianyi Zhou and Jing Xiao},
  journal={ArXiv},
  year={2023},
  volume={abs/2308.12032},
  url={https://api.semanticscholar.org/CorpusID:261076515}
}
```

