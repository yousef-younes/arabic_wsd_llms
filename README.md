# Arabic Word Sense Disambiguation with Large Language Models

## Overview

This repository contains the code and resources for experiments focused on Arabic Word Sense Disambiguation (WSD) using both small and large language models. The project explores different approaches to tackle WSD, including binary classification and multi-choice tasks, leveraging models such as mBERT and Llama. The experiments involve both zero-shot learning and fine-tuning methodologies.

## Project Structure

The repository is organized into the following main directories:

*   `bin_task/`: Contains notebooks and scripts related to binary classification WSD tasks.
*   `mc_task/`: Contains notebooks and scripts related to multi-choice WSD tasks.
*   `data/`: Stores the datasets used for training and evaluation.
*   `Results/`: Intended for storing experiment results and outputs.

## Experiments and Models

### Binary Classification WSD

The `bin_task` directory includes experiments that frame WSD as a binary classification problem. Key notebooks and scripts in this section are:

*   `Gemma_fine_tuning.ipynb`: This Jupyter Notebook demonstrates the fine-tuning process of the Google Gemma-2-27b model for Arabic WSD. It includes steps for data loading, preprocessing, model loading, and inference. The notebook utilizes the `transformers` library and `torch` for model operations. It also includes a prompt structure for zero-shot learning, where the model is instructed to output '1' if a candidate sense matches the word in context, and '0' otherwise.
*   `SLM_WSD_Binary.ipynb`: This notebook focuses on using small language models (SLMs) for binary WSD. It covers data preprocessing, tokenization using `BertTokenizer` (specifically `CAMeL-Lab/bert-base-arabic-camelbert-msa`), and preparing data for model training. It also includes code for setting up the environment for GPU usage.
*   `Open_LLMs_For_WSD_Binary.ipynb`: This notebook likely explores the application of other open-source Large Language Models for binary WSD tasks.
*   `WSD_Augmentation_with_Translation.ipynb`: This notebook suggests an approach to augment WSD data using translation, potentially to increase dataset size or introduce cross-lingual information.
*   `gemma_fine_tuning_exp.py`: A Python script version of the Gemma fine-tuning experiment, suitable for command-line execution.

{

  "word": "بنك",
  
  "context": "جلس الطفل على البنك بجانب النهر.",
  
  "gloss": "ضفة النهر",
  
  "label": 1
  
}


### Multi-Choice WSD

The `mc_task` directory contains experiments where WSD is approached as a multi-choice problem. The relevant files are:

*   `Closed_LLMs_MC.ipynb`: This notebook likely deals with the application of closed-source or proprietary Large Language Models for multi-choice WSD.
*   `Prepare_WSD_data_for_MCT.ipynb`: This notebook is dedicated to the preparation and formatting of WSD datasets specifically for multi-choice tasks.
*   `WSD_MC_FT.py`: A Python script for fine-tuning models on multi-choice WSD tasks.

{

  "word": "بنك",
  
  "context": "جلس الطفل على البنك بجانب النهر.",
  
  "candidates": [
  
    "مؤسسة مالية",
    
    "ضفة النهر",
    
    "مكان للتخزين"
    
  ],
  
  "label": 1
  
}

## Data

The `data/wsd/` directory contains JSON files (`bin_dev.json`, `bin_train.json`, `mc_dev.json`, `mc_train.json`) which are used for development and training of the WSD models in both binary and multi-choice settings. These files contain `context_id`, `context`, `word`, `sense`, `lemma_id`, `gloss_id`, and `label` (for binary tasks) or similar fields for multi-choice tasks.

## Setup and Usage

To set up the environment and run the experiments, follow these general steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yousef-younes/arabic_wsd_llms
    ```

2.  **Navigate to the repository directory:**

    ```bash
    cd arabic_wsd_llms
    ```

3.  **Install dependencies:**

    The notebooks indicate the use of `huggingface_hub`, `ipywidgets`, `transformers`, `accelerate`, `pandas`, `pyarabic`, `numpy`, and `torch`. You can install these using pip:

    ```bash
    pip install huggingface_hub ipywidgets transformers accelerate pandas pyarabic numpy torch
    ```

    *Note: Specific versions of libraries might be required based on the environment where the original experiments were run. Refer to the individual notebooks for precise dependency requirements.*

4.  **Run Jupyter Notebooks:**

    To run the `.ipynb` files, you will need Jupyter Notebook or JupyterLab. You can start it from the repository root:

    ```bash
    jupyter notebook
    ```

    Then navigate to the `bin_task` or `mc_task` directories and open the desired notebook.

5.  **Hugging Face Token:**

    Some notebooks (e.g., `Gemma_fine_tuning.ipynb`) require a Hugging Face token for authentication to access models. Ensure you have logged in with your token:

    ```python
    from huggingface_hub import login
    login(token="YOUR_HF_TOKEN") # Replace YOUR_HF_TOKEN with your actual token
    ```

## References

[1] KSAA-CAD: Contemporary Arabic Reverse Dictionary and Word Sense Disambiguation at ArabicNLP 2024. [https://github.com/ksaa-nlp/KSAA-CAD](https://github.com/ksaa-nlp/KSAA-CAD)
