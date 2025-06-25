# Superfiltering: Weak-to-Strong Data Filtering (ACL'24)

[Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning](https://arxiv.org/abs/2402.00530) (ACL'24)<br>
Chinese Version: [[知乎]](https://zhuanlan.zhihu.com/p/718119728)

<p align="center" width="40%">
<a ><img src="images/fast_alpaca.png" alt="overview" style="width: 40%; min-width: 300px; display: block; margin: auto;"></a>
</p>

This is the repo for the Superfiltering project, which introduces a method **astonishingly utilizes a small GPT2 (124M) model to successfully filter out the high-quality subset from the existing GPT4-generated instruction tuning dataset.**

The repo contains:

- The code for Superfiltering.
- The data selected by Superfiltering.
- The model checkpoints (7B) that were trained using our Superfiltering.

(This repo partially originated from [Cherry_LLM](https://github.com/MingLiiii/Cherry_LLM) and [Reflection_Tuning](https://github.com/tianyi-lab/Reflection_Tuning).)<br>
(Feel free to email Ming ([Homepage](https://mingliiii.github.io/), [Email](minglii@umd.edu)) for any questions or feedback.)

## News
- [2024/05] Our paper has been accepted to the **ACL 2024** main conference! 
- [2024/02] We added the codes and introduction for **Superfiltering with Diveristy** version, which can further compress the selected data to approximately **2%**. 
- [2024/02] We updated the repo of Superfiltering in which code and data were released. 
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
- [Our Related Works](#our-related-works)

## Overview

### Superfiltering

Instruction tuning is critical to improve LLMs but usually suffers from low-quality and redundant data. 
Data filtering for instruction tuning has proved important in improving both the efficiency and performance of the tuning process. 
But it also leads to extra cost and computation due to the involvement of LLMs in this process. 
To reduce the filtering cost, we study Superfiltering: Can we use a smaller and weaker model to select data for finetuning a larger and stronger model?
Despite the performance gap between weak and strong language models, we find their highly consistent capability to perceive instruction difficulty and data selection results. 
This enables us to use a much smaller and more efficient model to filter the instruction data used to train a larger language model. Not only does it largely speed up the data filtering, but the filtered-data-finetuned LLM achieves even better performance on standard benchmarks. 
Extensive experiments validate the efficacy and efficiency of our approach. 

<p align="center" width="50%">
<a ><img src="images/intro.png" alt="overview" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

**Top**: Comparison of data filtering for instruction tuning of a student model. (a) The filter model is a strong proprietary LLM, e.g. ChatGPT, which can be time-consuming and expensive but usually performs promisingly. (b) The filter model is the student model itself or a similar-sized open-source LLM, which is still time-consuming but free to use. (c) **Weak-to-strong Superfiltering** proposed by this paper, which utilizes a much smaller filter model, e.g. GPT-2, to train a stronger student LLM. We find it costs much less time but maintains the performance. <br>
**Bottom**: Comparisons of two student models finetuned using 5% data selected by LLaMA2-7B and GPT-2 from the Alpaca dataset. (d) Both models trained on 5% data outperform the baseline model trained on 100% data. (e) GPT-2 as the superfilter speeds up data filtering by 20 times. 

### Superfiltering with Diversity

Motivated by recent work that further includes Diversity metrics in the data selection process, we introduce an extended version of Superfiltering, **Superfiltering** with **D**iversity (**Superfiltering.D**). We hypothesize that the diversity metrics work better when implemented on a high-quality data subset than the whole dataset with mixed quality. Thus we propose to first utilize Superfiltering to select a subset with relatively high quality, then further utilize [Facility Location Function](https://apricot-select.readthedocs.io/en/latest/functions/facilityLocation.html#:~:text=Facility%20location%20functions%20are%20general,and%20their%20nearest%20chosen%20point.) to further compress the selected data number. Compared with other diversity metrics, the Facility Location Function can strike a balance between capturing diversity and ensuring the representation of different clusters or regions within the data, it ensures a global view of the given high-quality subset. To further preserve the efficiency of our Superfiltering.D, we utilize ```sentence-transformers/all-MiniLM-L6-v2``` as the encoder, which only has approximately 80M parameters. In our preliminary experiments on the Alpaca and Alpaca-GPT4 dataset, where we first select 20% of the data by Superfiltering, then utilize the Facility Location Function to further select 2% of the data. **The models trained with 2% of the data have a comparable or better performance than full data models.** 

The benefits of our **Superfiltering.D**:
1. We can compress the data selected to 2%, which further greatly improves the efficiency of Instruction Tuning.
2. This 2-step method, considering diversity only on the high-quality subset, relaxes the strong reliance on fancy encoders, ensuring that small encoders can work effectively.
3. This 2-step method greatly improves the efficiency of the diversity metrics, both the encoder and the diversity metric only need to compute on a subset rather than the whole great dataset.

## Highlights

* We reveal the **strong consistency between small and large LLMs in perceiving and evaluating the difficulty of instruction tuning data**, which provides insights into understanding the difference between small and large models. 
* We propose the first method of Superfiltering that utilizes **a small LM, e.g., GPT-2 (124M), to select data for instruction tuning and brings significant speedups to the LLM finetuning pipeline**. 
* Superfiltering is a **plug-and-play** method that precises in **allocating high-quality and informative data** improving LLM instruction tuning. 
* Our preliminary experiments show that by adding a simple diversity metric, our **Superfiltering.D** can use **only 2% of the data to defeat the full data model** on Alpaca and Alpaca-GPT4. 

## Install

Install the dependencies with `pip install -r requirements.txt`

Note: The calculation of IFD scores only needs the ```transformers``` package, thus if you are using a different code base with ```transformers``` installed, you can directly run the code and manually install the missing packages. 

## Run Code

### Superfiltering

1. Calculate IFD scores

```
bash scripts/step1_select_data_analysis_gpt2.sh
```

```--data_path```: The targeted dataset in the Alpaca format. <br>
```--save_path```: The path to save the ```.jsonl``` file containing scores. <br>
```--model_name_or_path```: The model used for calculating IFD scores, we found ```gpt2``` is good enough as illustrated in our paper. Also, you can use the model that you need to finetune, which would be a self-guided manner or student-involved manner. 

2. Put scores into the original data
```
bash scripts/step2_put_analysis_to_data.sh
```

```pt_data_path```: The ```.jsonl``` file generated in last step. <br>
```json_data_path```: The targeted dataset in the Alpaca format. <br>
```json_save_path```: The data path to save the data with IFD scores. <br>

Note: Steps 1 and 2 can be merged directly for better convenience. 

3. Select the data you wish. 
```
bash scripts/step3_select_data.sh
```

```json_data_path```: The data path to save the data with IFD scores. <br>
```json_save_path```: The data path to save the data with IFD scores filtered. <br>
```sample_rate```: How much data do you need? Here we only provide the percentage version, you can slightly modify the code to select the exact number you want. 

Note: The Step 1 code is the ```batch_size=1``` version, it takes about 15 minutes to process the whole Alpaca dataset. We release this version and split the whole process into 3 steps for better controllability. 
You can directly run the above 3 scripts to get a better understanding of our codes. 
It takes about 15 minutes for the whole process. 

### Superfiltering.D

To run Superfiltering.D, please first install the ```submodlib``` package [here](https://github.com/decile-team/submodlib).<br>
The step 1 and 2 are the same as the previous ones. 

3. Select the data with diversity.
```
scripts/optional_select_data_plus_diversity.sh
```

```json_data_path```: The data path to save the data with IFD scores. <br>
```json_save_path```: The data path to save the data with IFD scores filtered. <br>
```ifd_num```: The number of data you want for the high-quality subset, selected by the Superfiltering. <br>
```fla_num```: The number of data you want after implementing FacilityLocationFunction.

Note: In our preliminary experiments, setting ```ifd_num``` as 20% of the full data and ```fla_num``` as 2% of the full data works fine for both Alpaca and Alpaca-GPT4 datasets. <br>
Further experiments will be conducted. 

## Data

The Alpaca Data with GPT2-based IFD scores can be found in ```data/data_with_ifd/alpaca_data_gpt2_data.json```.<br>
The Alpaca-GPT4 Data with GPT2-based IFD scores can be found in ```data/data_with_ifd/alpaca_gpt4_data_gpt2_data.json```.<br>

To select the subset data from these datasets, you can directly run ```bash scripts/step3_select_data.sh``` in above Step 3. 

## Evaluation

The codes and data for pair-wise comparison by using GPT4 are released in the ```evaluation``` folder. 
This method greatly eliminates the potential position bias of GPT4 and chatGPT. 

To use this code, please follow the below scripts:

```bash evaluation/scripts/do_eval_generation.sh```: The model automatically generates the responses for a given instruction in test datasets. <br>
```bash evaluation/scripts/do_eval_generation_wrap.sh```: Wrap the response files of LLMs being compared. <br>
```bash evaluation/scripts/do_eval.sh```: Use GPT4 or chatGPT for the evaluation. <br>
```bash evaluation/scripts/do_review_eval_score.sh```: Parse the results and draw the figure. <be>

For other evaluation metrics, please see their official repo. 

## ToDo
- [x] Release the code, data, and models. 
- [x] Release Superfiltering with Diversity version
- [ ] Implement our method on more datasets and base models.  

## Citation

Please consider citing our papers if you think our codes, data, or models are useful. Thank you! <br>

```
@inproceedings{li-etal-2024-superfiltering,
    title = "Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning",
    author = "Li, Ming  and
      Zhang, Yong  and
      He, Shwai  and
      Li, Zhitao  and
      Zhao, Hongyu  and
      Wang, Jianzong  and
      Cheng, Ning  and
      Zhou, Tianyi",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.769",
    pages = "14255--14273",
}

@inproceedings{li-etal-2024-selective,
    title = "Selective Reflection-Tuning: Student-Selected Data Recycling for {LLM} Instruction-Tuning",
    author = "Li, Ming  and
      Chen, Lichang  and
      Chen, Jiuhai  and
      He, Shwai  and
      Gu, Jiuxiang  and
      Zhou, Tianyi",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.958",
    pages = "16189--16211",
}

@inproceedings{li-etal-2024-quantity,
  title = "From Quantity to Quality: Boosting {LLM} Performance with Self-Guided Data Selection for Instruction Tuning",
  author = "Li, Ming  and
    Zhang, Yong  and
    Li, Zhitao  and
    Chen, Jiuhai  and
    Chen, Lichang  and
    Cheng, Ning  and
    Wang, Jianzong  and
    Zhou, Tianyi  and
    Xiao, Jing",
  editor = "Duh, Kevin  and
    Gomez, Helena  and
    Bethard, Steven",
  booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
  month = jun,
  year = "2024",
  address = "Mexico City, Mexico",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2024.naacl-long.421",
  pages = "7595--7628",
}

@inproceedings{li2023reflectiontuning,
  title={Reflection-Tuning: Recycling Data for Better Instruction-Tuning},
  author={Ming Li and Lichang Chen and Jiuhai Chen and Shwai He and Tianyi Zhou},
  booktitle={NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following},
  year={2023},
  url={https://openreview.net/forum?id=xaqoZZqkPU}
}

```

## Our Related Works

If you are interested in **Data Selection** for Instruction Tuning, please see [Cherry_LLM](https://github.com/MingLiiii/Cherry_LLM) and [Superfiltering](https://github.com/tianyi-lab/Superfiltering). <br>
If you are interested in **human/LLM-free Data Augmentation** for Instruction Tuning, please see [Mosaic-IT](https://github.com/tianyi-lab/Mosaic-IT) and [RuleR](https://github.com/MingLiiii/RuleR). <br>
If you are interested in **Data Improvement** for Instruction Tuning, please see [Reflection_Tuning](https://github.com/tianyi-lab/Reflection_Tuning). <br>
If you are interested in **Knowledge Distillation** in the LLM era, please see this [Survey](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs). <br>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tianyi-lab/Superfiltering&type=Date)](https://star-history.com/#tianyi-lab/Superfiltering&Date)

