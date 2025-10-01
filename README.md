# EnviroGeno: Multimodal Transformer for Gene–Environment Interaction in Stroke Risk Prediction

This repository implements **EnviroGeno**, a multimodal Transformer-based framework designed to model **gene–environment interactions** in ischemic stroke (IS) risk prediction.

## Abstract

Genetic and environmental factors jointly influence ischemic stroke risk, yet existing predictive models either assume linear genetic effects or overlook genetic contributions when modelling environmental risk. More importantly, few frameworks capture the joint, non-linear interactions between genetic predispositions and modifiable environmental exposures. We propose EnviroGeno, a multi-modal Transformer-based deep learning framework that integrates dynamic environmental interaction through multi-head attention and conditional contrastive learning. EnviroGeno achieves the highest performance with an AUROC of 0.860 on the UK Biobank dataset. Risk stratification analyses demonstrated that individuals in the top 1% of predicted risk had a 2.17-fold increased stroke risk, and a 5-fold increase among those in the highest risk groups with smoking behaviour, family history and elevated BMI. Moreover, gene–environment interaction analyses identified variants such as CHRNA5 (smoking), TCF7L2 (type 2 diabetes and glucose metabolism), and ABCA1 (lipid metabolism) whose effects were amplified under adverse environmental exposures. External validation on the independent All of Us cohort (AUROC = 0.792) confirmed the model’s generalizability across diverse populations. These findings demonstrate that EnviroGeno enhances individualised stroke risk stratification, enabling clinicians to identify genetically susceptible individuals who may be overlooked by current clinical scores, and thereby support earlier interventions and personalised preventive strategies。

## Data Source

- **UK Biobank (UKBB):** [🔗 Apply for access](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access)  
- **All of Us Research Program:** [🔗 Research Hub](https://www.researchallofus.org/)  

This repository only includes dummy data to replicate the expected input format. To run with real data, apply for access through the official portals.

---

## Data Format

Expected input data (dummy format included):

- **Chromosome files:** `chr{i}.csv` (i = 1...22)  
  - Each file has shape `(N, S_i)` with SNPs encoded as **0, 1, 2**.  

- **Environmental features:** `environmental_data.csv`  
  - Includes continuous (e.g., BMI, sleep) and categorical (e.g., smoking, alcohol, family history, anxious feelings) features.
    
---

## Dependencies

- Python 3.8+  
- PyTorch >= 1.10
- NumPy >= 1.21  
- Pandas >= 1.3  
- scikit-learn >= 0.24  
- PyYAML  

Install via:
```bash
pip install -r requirements.txt
```
## Training

Run cross-validation training:
```bash
python main.py --config config.yaml
```


