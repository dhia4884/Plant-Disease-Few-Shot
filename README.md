# Few-Shot Plant Disease Recognition Using Self-Supervised Vision Foundation Models

## Research Project — Academic Year 2026–2027

**Research Area:** Artificial Intelligence and Data Science  
**Main Backbone:** DINOv2-small  
**Source Dataset:** PlantVillage  
**External Evaluation Dataset:** PlantDoc  
**Classification Approaches:** k-NN, Linear Probe, Prototype, and Prototype Refinement

---

## 1. Project Overview

This project investigates few-shot plant disease recognition using frozen visual representations extracted from a self-supervised vision foundation model.

The main objective is to examine whether prototype-based classification methods can improve recognition performance when only a limited number of labeled examples are available.

The research uses DINOv2-small as a frozen feature extractor and compares four classification approaches under 1-shot, 5-shot, and 10-shot learning settings.

PlantVillage is used for source-domain training, validation, and testing, while PlantDoc serves as an external field-domain evaluation dataset.

A particular emphasis is placed on evaluating generalization from controlled agricultural images to more challenging field conditions.

### Research Question

**Do prototype-based classifiers improve few-shot recognition of field plant diseases?**

---

## 2. Research Objectives

The main objectives are:

1. Investigate the use of self-supervised visual representations for few-shot plant disease recognition.
2. Evaluate frozen DINOv2-small embeddings as input features for lightweight classifiers.
3. Compare k-Nearest Neighbors, Linear Probe, Prototype Classification, and Prototype Refinement.
4. Study the effect of different labeled support-set sizes.
5. Evaluate generalization from PlantVillage to PlantDoc.
6. Analyze classification errors and per-class performance.
7. Investigate whether prototype refinement provides a reliable improvement over standard prototype classification.

---

## 3. Datasets

### 3.1 PlantVillage

PlantVillage is used as the primary source-domain dataset.

It provides labeled plant leaf images under relatively controlled imaging conditions.

The selected experimental subset contains 14 harmonized disease categories.

The dataset is divided into training, validation, and clean test subsets. A group-aware splitting strategy is used to reduce potential leakage between related leaf images.

### 3.2 PlantDoc

PlantDoc is used exclusively as an external field-domain evaluation dataset.

It contains images collected under more realistic agricultural conditions, including variations in backgrounds, illumination, and image quality.

The prepared external test subset contains:

- 109 images.
- 14 harmonized classes.
- Small and varying per-class supports.

The limited size of this subset is an important limitation and should be considered when interpreting the results.

### 3.3 Class Harmonization

A common 14-class label space is established to enable comparisons between the two datasets.

Original dataset labels are preserved in the metadata, while harmonized labels are used for the experiments.

The harmonization represents a practical label-matching strategy and does not imply that all source annotations are biologically identical.

---

## 4. Methodology

### 4.1 Feature Extraction

The pretrained `facebook/dinov2-small` model is used as a frozen feature extractor.

- Backbone: DINOv2-small.
- Input resolution: 224 × 224.
- Embedding dimension: 384.
- Backbone training: Disabled (frozen).

Image embeddings are extracted and stored separately from the classification experiments to avoid repeated feature extraction.

### 4.2 Few-Shot Learning Settings

The experiments use three support-set configurations:

| Setting | Labeled examples per class |
|---|---:|
| 1-Shot | 1 |
| 5-Shot | 5 |
| 10-Shot | 10 |

Three random seeds are used:

- 42
- 123
- 2026

Support sets are constructed from the training pool with group-level checks.

### 4.3 Classification Methods

Four approaches are evaluated:

**1. k-Nearest Neighbors (k-NN)**

A distance-based classifier operating on the extracted image embeddings.

**2. Linear Probe**

A Logistic Regression classifier trained on the few-shot support embeddings.

**3. Prototype Classifier**

Each class is represented by the mean embedding of its labeled support examples. Classification is performed using cosine similarity.

**4. Prototype Refinement**

An extension of the prototype classifier that incorporates selected unlabeled training-pool embeddings through confidence-based pseudo-labeling and prototype updates.

The refinement procedure uses a predefined cosine similarity threshold and limits the number of accepted pseudo-labels per class.

---

## 5. Experimental Protocol

The experimental workflow consists of:

1. Dataset inspection and preparation.
2. Class harmonization.
3. Duplicate and data leakage audits.
4. Group-aware training and validation splitting.
5. Frozen DINOv2 feature extraction.
6. Few-shot support-set construction.
7. Baseline classifier implementation.
8. Hyperparameter selection using validation data.
9. Prototype refinement experiments.
10. Locked final evaluation.
11. Per-class error analysis.
12. Statistical comparison and visualization.

Hyperparameters are selected using the validation subset. Final test results are not used for model selection.

The final evaluation considers both PlantVillage and PlantDoc across the three support-set sizes and three random seeds.

---

## 6. Evaluation Metrics

The following metrics and analyses are used:

- Macro F1-score.
- Classification accuracy.
- Mean and standard deviation across random seeds.
- Domain performance drop.
- Per-class performance.
- Confusion matrices.
- Exact McNemar test.
- Paired stratified bootstrap confidence intervals.

The domain performance drop is defined as:

\[
\text{Domain Drop}
=
F1_{\text{PlantVillage}}
-
F1_{\text{PlantDoc}}
\]

This quantity measures the absolute difference in Macro F1 between the source-domain and external-domain evaluations.

---

## 7. Main Experimental Findings

The final experiments show that classifier performance depends on both the number of labeled examples and the evaluation domain.

### 7.1 PlantVillage

Linear Probe achieves the strongest final-test Macro F1 across the three few-shot settings.

### 7.2 PlantDoc

Prototype Refinement achieves the highest mean Macro F1 in the 10-shot setting among the evaluated methods.

However, performance remains considerably lower than on PlantVillage, highlighting the difficulty of generalizing to field images.

### 7.3 Prototype Refinement

Prototype Refinement does not consistently outperform the standard Prototype classifier across all settings.

The paired statistical analysis does not establish a statistically reliable improvement.

All reported paired bootstrap 95% confidence intervals include zero, and the Holm-adjusted McNemar p-values do not indicate a significant difference.

These findings should be interpreted cautiously because the PlantDoc external test subset contains only 109 images and the experiments use three random seeds.

---

## 8. Project Structure

The repository is organized as follows:

```text
Plant-Disease-Few-Shot/
│
├── README.md
├── .gitignore
│
├── configs/
│   └── final_evaluation_config.json
│
├── docs/
│   └── PROMPT_HISTORY.md
│
├── notebook/
│   └── Plant_Disease_Few_Shot_Project.ipynb
│
├── metadata/
│   ├── plantdoc_external_test_manifest.csv
│   ├── plantvillage_test_manifest_clean.csv
│   ├── plantvillage_train_manifest.csv
│   ├── plantvillage_training_pool_manifest.csv
│   ├── plantvillage_validation_manifest.csv
│   └── few_shot_support_sets/
│       └── Nine support-set CSV files
│
├── results/
│   ├── final_evaluation/
│   │   ├── best_models_by_shots_and_domain.csv
│   │   ├── final_model_comparison.csv
│   │   ├── refinement_vs_prototype_comparison.csv
│   │   ├── per_class/
│   │   │   ├── confusion_matrices/
│   │   │   └── per-class evaluation CSV files
│   │   └── reproduced_predictions/
│   │       └── PlantDoc predictions for seeds 42, 123, and 2026
│   │
│   └── final_report_assets/
│       ├── domain_drop_comparison.csv
│       ├── final_model_comparison.csv
│       ├── mcnemar_statistical_results.csv
│       ├── paired_macro_f1_bootstrap.csv
│       └── Four evaluation figures
│
└── report/
    ├── Plant_Disease_Few_Shot_Report.pdf
    └── figures/
        └── Four report figures
```

The repository includes the research notebook, experimental configuration, dataset manifests, few-shot support sets, selected evaluation results, statistical analysis outputs, visualizations, and the final PDF report.

Original image datasets and extracted feature embeddings are not included.

---

## 9. Data and Artifact Management

The original image datasets and extracted DINOv2 embeddings are stored separately in Google Drive.

Large feature matrices and original image collections are not required to be committed to this repository.

The repository focuses on research documentation, metadata, experimental configurations, selected results, visualizations, and reporting materials.

The metadata files contain references to image locations and experiment identifiers. Their original paths may need to be adapted when reproducing the experiments in another environment.

---

## 10. Reproducibility

The experiments were developed using Google Colab.

The main research workflow is maintained in a single Jupyter Notebook.

To reproduce the experiments, the user needs:

1. Access to the prepared datasets.
2. Access to the saved DINOv2 embeddings or the ability to extract them again.
3. The corresponding metadata and support-set files.
4. The final evaluation configuration.
5. A compatible Python environment and the required machine-learning libraries.

The final configurations and saved results are provided to support experiment traceability.

Exact reproduction may require restoring the original Google Drive directory structure and software environment.

---

## 11. Limitations

Several limitations should be considered:

- The PlantDoc external evaluation contains only 109 images.
- Class supports are small and vary between categories.
- Only three random seeds are used.
- Label harmonization between datasets may introduce uncertainty.
- Group-based auditing reduces leakage risks but cannot conclusively eliminate all possible overlap among unmatched records.
- The prototype refinement normalization ablation was not performed separately.
- The confidence threshold used for pseudo-label selection is a similarity threshold, not a calibrated probability.
- The results do not establish that prototype refinement is universally superior.

These limitations motivate further experiments using larger field datasets, additional random seeds, and more extensive ablation studies.

---

## 12. Responsible Use of AI

AI assistance was used as a supporting tool for research planning, literature organization, implementation guidance, debugging, experimental design, result interpretation, visualization planning, and scientific report preparation.

Representative reconstructed prompts are provided in:

`docs/PROMPT_HISTORY.md`

The prompt history is a reconstructed summary rather than a verbatim transcript of the original interactions.

The researcher remains responsible for verifying sources, understanding the methodology, checking the experimental results, and defending the scientific conclusions.

---

## 13. Repository

**GitHub:**  
https://github.com/dhia4884/Plant-Disease-Few-Shot

---

## 14. Acknowledgements and References

The research builds upon publicly available datasets and pretrained visual representation models.

Relevant publications and dataset references should be documented in the accompanying scientific report and its bibliography.


