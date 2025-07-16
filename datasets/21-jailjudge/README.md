---
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - JAILJUDGE_TRAIN.json
  - split: test
    path:
    - JAILJUDGE_ID.json
    - JAILJUDGE_OOD.json
size_categories:
- 10K<n<100K
license: other
license_name: jailjudge
license_link: LICENSE
task_categories:
- text-classification
- question-answering
- text-generation
language:
- en
- zh
- it
- vi
- ar
- ko
- th
- bn
- sw
- jv
---


## Overview

Although significant research efforts have been dedicated to enhancing the safety of large language models (LLMs) by understanding and defending against jailbreak attacks, evaluating the defense capabilities of LLMs against jailbreak attacks  also attracts lots of attention. Current evaluation methods lack explainability and do not generalize well to complex scenarios, resulting in incomplete and inaccurate assessments (e.g., direct judgment without reasoning explainability,  the F1 score of the GPT-4 judge is only 55\% in complex scenarios and bias evaluation on multilingual scenarios, etc.). To address these challenges, we have developed a comprehensive evaluation benchmark, JAILJUDGE, which includes a wide range of risk scenarios with complex malicious prompts (e.g., synthetic, adversarial, in-the-wild, and multi-language scenarios, etc.) along with high-quality human-annotated test datasets. Specifically, the JAILJUDGE dataset comprises training data of JAILJUDGE, with over 35k+ instruction-tune training data with reasoning explainability, and JAILJUDGETEST, a 4.5k+ labeled set of broad risk scenarios and a 6k+ labeled set of multilingual scenarios in ten languages. To provide reasoning explanations (e.g., explaining why an LLM is jailbroken or not) and fine-grained evaluations (jailbroken score from 1 to 10), we propose a multi-agent jailbreak judge framework, JailJudge MultiAgent, making the decision inference process explicit and interpretable to enhance evaluation quality.   Using this framework, we construct the instruction-tuning ground truth and then instruction-tune an end-to-end jailbreak judge model, JAILJUDGE Guard, which can also provide reasoning explainability with fine-grained evaluations without API costs. 
Additionally, we introduce JailBoost, an attacker-agnostic attack enhancer, and GuardShield, a safety moderation defense method, both based on JAILJUDGE Guard. Comprehensive experiments demonstrate the superiority of our JAILJUDGE benchmark and jailbreak judge methods. Our jailbreak judge methods (JailJudge MultiAgent and JAILJUDGE Guard) achieve SOTA performance in closed-source models (e.g., GPT-4) and safety moderation models (e.g., Llama-Guard and ShieldGemma, etc.), across a broad range of complex behaviors (e.g., JAILJUDGE benchmark, etc.) to zero-shot scenarios (e.g., other open data, etc.). Importantly, JailBoost and GuardShield, based on JAILJUDGE Guard, can enhance downstream tasks in jailbreak attacks and defenses under zero-shot settings with significant improvement (e.g., JailBoost can increase the average performance  by approximately 29.24\%, while GuardShield can reduce the average defense ASR from 40.46\% to 0.15\%).

## 💡Framework


The JAILJUDGE Benchmark encompasses a wide variety of complex jailbreak scenarios, including multilingual and adversarial prompts, targeting diverse LLM responses for robust safety evaluation.

The JAILJUDGE Data includes over 35k instruction-tune training data and two test sets (4.5k+ broad risk scenarios and 6k+ multilingual examples), providing a rich foundation for comprehensive jailbreak assessments.

The Multi-agent Jailbreak Judge Framework leverages multiple agents (Judging, Voting, and Inference agents) to deliver fine-grained evaluations, reasoning explanations, and jailbroken scores, making the evaluation process explicit and interpretable.


asets

## 👉 Paper
For more details, please refer to our paper [JAILJUDGE](https://arxiv.org/abs/2410.12855).


# Dataset Card for Dataset Name

<!-- Provide a quick summary of the dataset. -->


# Dataset
We open-sourced two comprehensive datasets, **JAILJUDGETRAIN** and **JAILJUDGETEST**, designed to advance research in the jailbreak evaluation of LLMs. Both datasets encompass a wide range of prompt-response pairs and are structured to support jailbreak evaluation  methods. To assist researchers in replicating our work,  The complete datasets and supporting resources are available on [Hugging Face JAILJUDGE Dataset](https://huggingface.co/datasets/ICLR-Anonymous/JAILJUDGE)


### JAILJUDGETRAIN Dataset

The **JAILJUDGETRAIN** dataset is a comprehensive instruction-tuning dataset derived using our multi-agent jailbreak judge framework, specifically designed to evaluate the jailbreak potential of LLMs. This large-scale dataset consists of over 35,000 items, gathered from diverse sources to capture a wide range of risk and response scenarios. The framework employs multiple LLM-powered agents that collaboratively assess each prompt-response pair, providing detailed scoring and reasoning for each instance.

Each data instance in **JAILJUDGETRAIN** includes the following fields:
- **user_prompt**: The prompt given to the model.
- **model_response**: The model's response to the prompt.
- **hazard_cate_llamaguard3**: The hazard category of the prompt, classified by the LlamaGuard3 model.
- **prompt_complexity_cate**: The complexity category of the prompt, indicating the sophistication level of the input.
- **language**: The language in which the prompt is written.
- **reason**: The explanation or rationale for the score assigned to the model's response, providing insights into why the response may or may not indicate a jailbreak.
- **score**: A numerical score that represents the degree to which the model’s response aligns with jailbreak behavior, as evaluated by the multi-agent framework.

### Prompt Categories in JAILJUDGETRAIN

The dataset contains six distinct categories of prompts, covering a comprehensive range of risk scenarios and complexities:
1. **Vanilla Harmful Prompts**: These include a broad array of risk scenarios.
2. **Synthetic Vanilla Prompts**: Rewritten by LLMs to preserve original meanings with slight modifications.
3. **Synthetic Adversarial Prompts**: Modified prompts created to simulate jailbreak attacks.
4. **In-the-Wild Harmful Prompts**: Real-world user interactions with LLMs.
5. **Deceptive Harmful Prompts**: Complex and refined prompts, automatically generated to challenge jailbreak detection.

### Purpose and Utility

This multi-agent setup not only enhances the accuracy of jailbreak assessment but also ensures a thorough understanding of the reasoning behind each decision. With structured scores and detailed explanations, the **JAILJUDGETRAIN** dataset is a valuable resource for training and fine-tuning models focused on jailbreak detection tasks.


### JAILJUDGETEST
JAILJUDGETEST is a high-quality, human-annotated evaluation set consisting of two subsets: **JAILJUDGE ID** (in-distribution) and **JAILJUDGE OOD** (out-of-distribution). The JAILJUDGE ID set includes over 4,500 prompt-response pairs from JAILJUDGETRAIN (excluding multilingual prompts) and is balanced for various risk scenarios. The JAILJUDGE OOD set, focused on multilingual prompts across ten languages, includes over 6,000 instances. This OOD set is specifically designed to assess the model’s generalizability to diverse linguistic contexts.

### Human Annotation and Quality Assurance
To ensure high-quality labels, we employed a rigorous, multi-stage annotation process:
1. **Annotator Training**: Following MLCommons standards, annotators were trained on policies regarding user privacy, ethics, and morality.
2. **Manual Labeling and GPT-4 Assistance**: Initial labeling was performed manually, with GPT-4 providing secondary labels. Discrepancies between human and GPT-4 annotations were reviewed to ensure accuracy.
3. **Cross-Comparison and Multi-Person Voting**: Final labels were determined by majority voting, involving multiple annotators to increase reliability. 

### Prompt Complexity Categories
We categorize prompts into five complexity levels:
1. **Simple Prompts (Q1)**: Direct, straightforward prompts.
2. **Adversarial Prompts (Q2)**: Generated by jailbreak attacks.
3. **In-the-Wild Prompts (Q3)**: Real-world prompts with adversarial elements.
4. **Synthetic Vanilla Prompts (Q4)**: Rephrased prompts that retain original meanings.
5. **Deceptive Harmful Prompts (Q5)**: Complex, sophisticated prompts combining multiple adversarial techniques.


###  Hazard Categories for  Harmful Prompts

The following table provides a summary of the **hazard categories** for harmful prompts in the JAILJUDGE dataset. Each category code represents a specific type of harmful or sensitive content that the prompts aim to address.

| Category Code | Description                  | Category Code | Description                  |
|---------------|------------------------------|---------------|------------------------------|
| S1            | Violent Crimes               | S8            | Intellectual Property        |
| S2            | Non-Violent Crimes           | S9            | Indiscriminate Weapons       |
| S3            | Sex-Related Crimes           | S10           | Hate                          |
| S4            | Child Sexual Exploitation    | S11           | Suicide & Self-Harm          |
| S5            | Defamation                   | S12           | Sexual Content               |
| S6            | Specialized Advice           | S13           | Elections                    |
| S7            | Privacy                      | S14           | Code Interpreter Abuse       |




# Dataset Card for JAILJUDGE Dataset

## Dataset Summary

The JAILJUDGE dataset is a comprehensive collection of prompt-response pairs specifically curated to evaluate the jailbreak potential of LLMs across multiple scenarios. The dataset is divided into three main files, each with unique attributes to facilitate both in-distribution (ID) and out-of-distribution (OOD) testing. This dataset was constructed using our multi-agent jailbreak judge framework, which enables precise scoring, reasoning, and categorization of each response.

## Files and Fields

### JAILJUDGE_TRAIN.json
- **user_prompt**: The prompt given to the model.
- **model_response**: The model's response to the prompt.
- **hazard_cate_llamaguard3**: The hazard category of the prompt, as classified by the LlamaGuard3 model.
- **prompt_complexity_cate**: The complexity category of the prompt, indicating its sophistication.
- **language**: The language in which the prompt is written.
- **reason**: The rationale or explanation for the score given to the model's response.
- **score**: A numerical score (1-10) indicating the likelihood of jailbreak behavior in the response.

### JAILJUDGE_ID.json
- **user_prompt**: The prompt given to the model.
- **model_response**: The model's response to the prompt.
- **hazard_cate_llamaguard3**: The hazard category of the prompt, as classified by the LlamaGuard3 model.
- **prompt_complexity_cate**: The complexity category of the prompt.
- **language**: The language of the prompt.
- **is_jailbroken**: Boolean value (True or False) indicating whether the model's response was classified as jailbroken.

### JAILJUDGE_OOD.json
- **user_prompt**: The prompt given to the model.
- **model_response**: The model's response to the prompt.
- **hazard_cate_llamaguard3**: The hazard category of the prompt, as classified by the LlamaGuard3 model.
- **prompt_complexity_cate**: The complexity category of the prompt.
- **language**: The language of the prompt.
- **is_jailbroken**: Boolean value (True or False) indicating whether the model's response was classified as jailbroken.

## Detailed Field Descriptions

- **prompt_complexity_cate**: Represents the complexity level of the prompt. The categories are as follows:
  - **Simple Prompts (Q1)**: Direct and straightforward prompts.
  - **Adversarial Prompts (Q2)**: Prompts generated by jailbreak attacks.
  - **In-the-Wild Prompts (Q3)**: Prompts sourced from real-world user interactions.
  - **Synthetic Vanilla Prompts (Q4)**: Rephrased prompts that retain their original meanings.
  - **Deceptive Harmful Prompts (Q5)**: Sophisticated prompts combining multiple adversarial elements.

- **hazard_cate_llamaguard3**: The hazard category of the prompt, based on classifications from the LlamaGuard3 model. This field categorizes prompts into various risk types, such as:
  - **S1**: Violent Crimes
  - **S2**: Non-Violent Crimes
  - **S3**: Sex-Related Crimes
  - **S4**: Child Sexual Exploitation
  - **S5**: Defamation
  - **S6**: Specialized Advice
  - **S7**: Privacy
  - **S8**: Intellectual Property
  - **S9**: Indiscriminate Weapons
  - **S10**: Hate
  - **S11**: Suicide & Self-Harm
  - **S12**: Sexual Content
  - **S13**: Elections
  - **S14**: Code Interpreter Abuse

- **language**: Indicates the language in which the prompt is written. Supported languages include:
  - **en**: English
  - **zh**: Chinese
  - **it**: Italian
  - **vi**: Vietnamese
  - **ar**: Arabic
  - **ko**: Korean
  - **th**: Thai
  - **bn**: Bengali
  - **sw**: Swahili
  - **jv**: Javanese

- **is_jailbroken**: A Boolean value (True or False) that specifies whether the model's response to the prompt was classified as jailbroken.

- **score**: A numerical score ranging from 1 to 10 that indicates the extent to which the model's response is deemed as exhibiting jailbreak behavior, with higher scores representing a greater likelihood of jailbreak.

- **reason**: A detailed explanation justifying the score assigned to the model's response, offering insights into why the response may or may not be considered jailbroken.

## Usage Notes

The JAILJUDGE dataset provides a valuable resource for training and evaluating jailbreak detection models. The structured and annotated fields enable researchers to study various risk scenarios, prompt complexities, and response characteristics across multiple languages and hazard categories.


##  Citation
If you find this project helpful, please consider citing our paper:
```
@misc{liu2024jailjudgecomprehensivejailbreakjudge,
      title={JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework}, 
      author={Fan Liu and Yue Feng and Zhao Xu and Lixin Su and Xinyu Ma and Dawei Yin and Hao Liu},
      year={2024},
      eprint={2410.12855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.12855}, 
}
```
