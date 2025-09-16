# MSAQE: Multi-view Scenic Areas Quality Evaluation

![11111111111_00](https://github.com/user-attachments/assets/204efa0f-7605-4111-a379-a5b5adcd1ed4)

This repository hosts the implementation of the **Multi-view Scenic Areas Quality Evaluation (MSAQE)** framework, which assesses scenic area quality through user-generated content. The framework integrates two primary components:

This article has been accepted by DASFAA 2025.

1. **Reference-Based Sentiment Analysis (RBSA):** Build reference set. Utilizes a fine-tuned ALBERT model to generate sentiment scores for comments, enhancing accuracy by comparing them to a curated reference set.
2. **Global and Local Ensemble Encoding (GLEE):** Employs GLEE for multi-label classification, capturing various quality dimensions such as business management (BM.), excursions (Exc.), hygiene (Hyg.), post and telecommunications (PT.), tourism transportation (TT.), travel safety (TSA.), travel shopping (TSH.), and resources and environmental protection (REP.).



## Repository Structure
```
├── code/
│   ├── CBloss_bigbird_GLEE_atten_kernel/
│   │   ├── BertCNNClassifier.py               # BERT-based CNN classifier
│   │   ├── BertCNNClassifier_att.py           # BERT-based CNN classifier with attention
│   │   ├── CBloss_bigbird_Just_Label_202...   # Model implementation for bigbird
│   │   ├── CBloss_bigbird_Just_Label_att.py   # Model with attention mechanism
│   │   ├── config.yml                         # Configuration file
│   │   ├── test.py                            # Testing script
│   │   ├── util_loss.py                       # Utility functions for loss computation
│   ├── RBSA/
│   │   ├── average_calculate copy.py          # Calculation script (copy version)
│   │   ├── calculate_albert_and_RBSA_sen...   # Sentiment calculation using ALBERT & RBSA
│   │   ├── calculate_albert_sentiment_fro...  # Sentiment calculation from ALBERT
│   │   ├── calculate_similarity_from_tests... # Similarity calculation
│   ├── roberta_chinese_GLEE_atten_kernel/
│   │   ├── BertCNNClassifier.py               # BERT-based CNN classifier
│   │   ├── BertCNNClassifier_att.py           # BERT-based CNN classifier with attention
│   │   ├── Roberta_Just_Label_20230526.py     # Model implementation for RoBERTa
│   │   ├── Roberta_Just_Label_att.py          # RoBERTa model with attention mechanism
│   │   ├── config.yml                         # Configuration file
│   │   ├── test.py                            # Testing script
│   │   ├── util_loss.py                       # Utility functions for loss computation
│   ├── roformer_v2_chinese_char_base_GLEE_atten_kernel/
│   │   ├── BertCNNClassifier.py               # BERT-based CNN classifier
│   │   ├── BertCNNClassifier_att.py           # BERT-based CNN classifier with attention
│   │   ├── Roformer_v2_Just_Label_20230...    # Model implementation for RoFormer
│   │   ├── Roformer_v2_Just_Label_att.py      # RoFormer model with attention mechanism
│   │   ├── config.yml                         # Configuration file
│   │   ├── test.py                            # Testing script
│   │   ├── util_loss.py                       # Utility functions for loss computation
├── README.md                                  # Project documentation
├── requirements.txt                           # Required Python packages
├── testset.csv                                # Test dataset
├── valset.csv                                 # Validation dataset
```



---

## Features

- **Sentiment Analysis with RBSA:** Leverages ALBERT and curated reference datasets for precise sentiment scoring.  
- **Fine-grained Multi-label Classification with GLEE:** Employs global and local encoding with attention mechanisms for accurate scenic area quality evaluation.  
- **Custom Datasets:** Includes sample datasets for easy testing and validation.

---

## Getting Started

### 1. Clone the repository

Clone the repository from GitHub:

```bash
git clone https://github.com/Yajie-good/MSAQE
```

### 2. Navigate to the project directory
Move into the project root directory:
```bash
cd MSAQE
```

### 3. Install dependencies
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

### 4. Explore the code
For sentiment analysis, navigate to 
```bash
code/RBSA/
```
For multi-label classification, navigate to 
```bash
code/roformer_v2_chinese_char_base_GLEE_atten_kernel_1_3/
code/roberta_chinese_GLEE_atten_kernel_1_2/
code/CBloss_bigbird_GLEE_atten_kernel_2_3/
```

### Datasets

Train Dataset: 
```bash
download from  https://pan.baidu.com/s/1DeQvjGt5q_muJh3-VzAGNQ?pwd=6666 code:6666
```

Validation Dataset: 
```bash
valset.csv
```
Test Dataset:
```bash
 testset.csv
```

