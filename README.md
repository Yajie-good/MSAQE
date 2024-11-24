# MSAQE: Multi-view Scenic Areas Quality Evaluation
![Uploading image.png…]()


This repository hosts the implementation of the **Multi-view Scenic Areas Quality Evaluation (MSAQE)** framework, which assesses scenic area quality through user-generated content. The framework integrates two primary components:

1. **Reference-Based Sentiment Analysis (RBSA):** Build reference set. Utilizes a fine-tuned ALBERT model to generate sentiment scores for comments, enhancing accuracy by comparing them to a curated reference set.
2. **Global and Local Ensemble Encoding (GLEE):** Employs GLEE for multi-label classification, capturing various quality dimensions such as business management (BM.), excursions (Exc.), hygiene (Hyg.), post and telecommunications (PT.), tourism transportation (TT.), travel safety (TSA.), travel shopping (TSH.), and resources and environmental protection (REP.).



## Repository Structure
├── code/ │ ├── RBSA/ │ │ ├── albert_fine_tuning.py # Fine-tuning ALBERT for sentiment analysis │ │ ├── reference_dataset.py # Construction and management of reference datasets │ │ ├── sentiment_scoring.py # Calculation of sentiment scores for new comments │ ├── roformer_v2_chinese_char_base_GLEE_atten_kernel_1_3/ │ │ ├── model.py # Implementation of the GLEE framework with attention │ │ ├── training_pipeline.py # Training and evaluation of the multi-label classifier │ │ ├── ablation_study.py # Code for ablation studies (e.g., kernel size effects) ├── data/ # Contains sample datasets for testing ├── README.md # Project documentation ├── requirements.txt # Required Python packages ├── valset.csv # Validation dataset ├── testset.csv # Test dataset


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
git clone https://anonymous.4open.science/r/MSAQE
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
```

### Datasets
Validation Dataset: 
```bash
valset.csv
```
Test Dataset:
```bash
 testset.csv
```

