# Reconstructed Representative Prompt History
## Few-Shot Plant Disease Recognition Using Self-Supervised Vision Foundation Models



---

## 1. Research Topic Selection and Project Planning

### Prompt 1 — Research Topic Exploration

> I need to develop a research project related to computer vision and few-shot learning.
>
> Propose research directions involving modern vision foundation models, self-supervised learning, and agricultural image analysis. Prioritize topics that are scientifically meaningful, feasible within a limited academic timeframe, and supported by publicly available datasets.
>
> For each direction, explain its research motivation, potential contribution, technical difficulty, required resources, and possible evaluation strategy.

### Prompt 2 — Defining the Research Scope

> Help me formulate a focused research project entitled "Few-Shot Plant Disease Recognition Using Self-Supervised Vision Foundation Models."
>
> Define a precise research question, research objectives, hypotheses, experimental scope, and expected contributions.
>
> Keep the project feasible within approximately six weeks. Avoid unnecessary model complexity and prioritize a reproducible experimental design that can be clearly explained and defended academically.

### Prompt 3 — Experimental Roadmap

> Design a complete step-by-step research workflow for this project, starting from literature review and dataset preparation and ending with final evaluation, error analysis, and scientific reporting.
>
> Organize the workflow into logical phases. For every phase, specify its objective, required inputs, expected outputs, and validation checks.
>
> Ensure that the experimental design distinguishes between model development, validation, and final testing.

---

## 2. Literature Review and Scientific Background

### Prompt 4 — Literature Review Strategy

> Identify the main research areas and foundational concepts relevant to few-shot plant disease recognition using self-supervised vision foundation models.
>
> Organize the literature review around:
>
> 1. Plant disease recognition using computer vision.
> 2. Limitations of conventional supervised learning.
> 3. Few-shot learning and low-data classification.
> 4. Self-supervised visual representation learning.
> 5. Vision Transformers and DINOv2.
> 6. Prototype-based classification methods.
> 7. Domain generalization and the difference between controlled and field images.
>
> Suggest relevant primary papers, official documentation, and dataset publications. Distinguish established findings from potential research gaps.

### Prompt 5 — Research Motivation and Introduction

> Draft a scientifically structured introduction for the research project.
>
> Explain why plant disease recognition is challenging, why collecting labeled agricultural images can be difficult, and how few-shot learning may address the limited-data problem.
>
> Introduce self-supervised vision foundation models, particularly DINOv2, as feature extractors.
>
> Clearly explain the motivation for evaluating prototype-based classifiers and testing their generalization from PlantVillage to PlantDoc.
>
> Use cautious academic language and avoid unsupported claims.

### Prompt 6 — Dataset and Source Verification

> Investigate the suitability of PlantVillage and PlantDoc for this research.
>
> Describe their origins, characteristics, image domains, labeling structures, and known limitations.
>
> Identify their official or primary distribution sources and relevant licensing information.
>
> Explain why PlantVillage can be used for source-domain experiments while PlantDoc can serve as an external field-domain evaluation dataset.
>
> Highlight potential concerns related to image duplication, label compatibility, and copyright.

---

## 3. Dataset Preparation and Data Integrity

### Prompt 7 — Class Harmonization

> Develop a consistent class-mapping strategy between PlantVillage and PlantDoc.
>
> Select disease categories that can be reasonably matched across both datasets and define a unified 14-class label space.
>
> Provide a mapping table that preserves the original dataset labels while assigning a common class identifier.
>
> Explain any ambiguous mappings and avoid assuming that visually similar labels necessarily represent identical biological conditions.

### Prompt 8 — Dataset Integrity and Leakage Prevention

> Design a data integrity audit for the selected PlantVillage and PlantDoc images.
>
> Include checks for missing files, corrupted images, duplicate image hashes, conflicting labels, and possible overlap between training and testing data.
>
> Explain how group-aware splitting can reduce the risk of placing related leaf images in both training and validation subsets.
>
> Define a conservative policy for handling suspicious duplicates and document any remaining limitations.

### Prompt 9 — Dataset Manifests and Reproducibility

> Propose a reproducible manifest structure for the prepared datasets.
>
> The manifests should preserve image paths, class labels, group identifiers, and the information needed to reconstruct the experiments.
>
> Separate the PlantVillage training pool, validation subset, clean test subset, and PlantDoc external test subset.
>
> Include automated assertions to verify class distributions, image availability, group separation, and consistency between labels and class mappings.

---

## 4. Feature Extraction Using DINOv2

### Prompt 10 — Frozen Feature Extraction

> Implement a feature extraction pipeline using the pretrained facebook/dinov2-small model.
>
> Use the official Hugging Face Transformers interface and extract 384-dimensional image embeddings.
>
> Keep the backbone frozen and use a consistent image preprocessing pipeline with a 224-pixel input resolution.
>
> The implementation should support GPU execution in Google Colab, process images in batches, and save embeddings, labels, class names, and extraction configuration files.
>
> Include checks for invalid embeddings, unexpected tensor shapes, and reproducibility.

### Prompt 11 — Feature Extraction Validation

> Design validation checks for the extracted PlantVillage and PlantDoc embeddings.
>
> Confirm that the feature matrices have the expected dimensions, labels correspond to the correct class ordering, and all numerical values are finite.
>
> Explain why extracting embeddings once and reusing them across classifier experiments improves experimental efficiency and reproducibility.

---

## 5. Few-Shot Experimental Design

### Prompt 12 — Support Set Construction

> Design few-shot support sets for 14 classes using 1, 5, and 10 labeled examples per class.
>
> Use three fixed random seeds: 42, 123, and 2026.
>
> Ensure that selected support examples originate exclusively from the training pool and that distinct leaf groups are selected within each class.
>
> Save the support-set manifests and implement checks for exact per-class counts, duplicate paths, and group-level separation.

### Prompt 13 — Baseline Classifiers

> Implement and evaluate three few-shot classification approaches using the frozen DINOv2 embeddings:
>
> 1. k-Nearest Neighbors.
> 2. Linear Probe using Logistic Regression.
> 3. Prototype-based classification using class mean embeddings and cosine similarity.
>
> Evaluate 1-shot, 5-shot, and 10-shot settings across the predefined random seeds.
>
> Tune hyperparameters using only the validation subset. Report Macro F1, accuracy, and variability across seeds.
>
> Keep the implementation modular and save experiment configurations and results.

### Prompt 14 — Hyperparameter Selection

> Establish a fair validation protocol for the three baseline classifiers.
>
> For k-NN, evaluate a reasonable range of neighborhood sizes and embedding normalization settings.
>
> For Logistic Regression, evaluate several regularization strengths and normalization settings.
>
> For prototype classification, compare normalized and unnormalized embeddings.
>
> Select configurations using validation Macro F1 and freeze the selected configurations before final test evaluation.
>
> Do not use the final test labels to select hyperparameters.

---

## 6. Prototype Refinement Method

### Prompt 15 — Refinement Method Design

> Develop a simple prototype refinement procedure using unlabeled examples from the PlantVillage training pool.
>
> Begin with prototypes computed from the labeled few-shot support set.
>
> Assign pseudo-labels to sufficiently confident unlabeled examples using cosine similarity and a predefined confidence threshold.
>
> Limit the number of pseudo-labeled examples per class and update each prototype using a weighted combination of its original representation and the pseudo-labeled centroid.
>
> Ensure that the refinement procedure does not access the ground-truth labels of the unlabeled pool when assigning pseudo-labels.
>
> Explain the algorithm, its assumptions, and its potential failure modes.

### Prompt 16 — Refinement Validation and Ablation

> Evaluate the prototype refinement method under the same few-shot settings and random seeds used for the baseline classifiers.
>
> Compare its validation performance against the original prototype classifier.
>
> Report the number of accepted pseudo-labels and investigate whether refinement consistently improves classification performance.
>
> Clearly distinguish between experiments that were performed and ablation experiments that were not performed.

---

## 7. Final Evaluation and Domain Generalization

### Prompt 17 — Locked Final Evaluation

> Prepare the final evaluation protocol for all four classifiers: k-NN, Linear Probe, Prototype, and Prototype Refinement.
>
> Evaluate each model using the locked configurations selected during validation.
>
> Test on both the clean PlantVillage test subset and the independent PlantDoc external test subset.
>
> Report Macro F1, accuracy, mean performance, and standard deviation across the three predefined seeds.
>
> Do not modify model configurations based on final test results.

### Prompt 18 — Domain Generalization Analysis

> Analyze the difference between performance on PlantVillage and PlantDoc.
>
> Calculate the absolute domain performance drop as:
>
> Domain Drop = PlantVillage Macro F1 - PlantDoc Macro F1.
>
> Compare the four classifiers across 1-shot, 5-shot, and 10-shot settings.
>
> Discuss the possible effects of differences in image backgrounds, lighting, visual complexity, and dataset size.
>
> Avoid claiming that a method generalizes better solely from one metric or one experimental setting.

---

## 8. Error Analysis and Statistical Evaluation

### Prompt 19 — Per-Class Performance Analysis

> Perform a detailed per-class evaluation of the four classifiers, with particular attention to the 10-shot setting.
>
> Identify the most difficult and easiest PlantDoc classes.
>
> Compare per-class Macro F1 values and investigate which categories benefit from prototype refinement and which experience performance degradation.
>
> Use the available confusion matrices to identify recurring classification errors.
>
> Interpret results cautiously because the external test set contains only 109 images.

### Prompt 20 — Confusion Matrix Analysis

> Generate and analyze confusion matrices for all four classifiers on both evaluation domains.
>
> Validate that each confusion matrix has the expected 14-by-14 shape and that its total number of predictions matches the corresponding test-set size.
>
> Produce clear visual comparisons and identify important confusion patterns without overstating conclusions from small class supports.

### Prompt 21 — Paired Statistical Comparison

> Reconstruct per-image predictions for the 10-shot PlantDoc experiments using the saved embeddings, support sets, and locked configurations.
>
> First verify that reconstructed confusion matrices exactly match the original saved confusion matrices.
>
> Then compare Prototype and Prototype Refinement using exact McNemar tests for paired classification correctness and a paired stratified bootstrap for Macro F1 differences.
>
> Report confidence intervals, p-values, and appropriate multiple-comparison corrections.
>
> Interpret the results conservatively and do not claim statistical significance when the evidence does not support it.

---

## 9. Scientific Interpretation and Reporting

### Prompt 22 — Results and Discussion

> Write a scientific interpretation of the final experimental results.
>
> Discuss the differences between controlled-domain and field-domain performance, the effect of increasing the number of labeled examples, and the relative behavior of the four classifiers.
>
> Explain why prototype refinement may help in some settings but fail to improve others.
>
> Address the limitations of the small PlantDoc test set, the limited number of random seeds, and the absence of a separate refinement-normalization ablation.
>
> Ensure that every conclusion is supported by the reported results.

### Prompt 23 — Limitations and Future Work

> Prepare a dedicated limitations and future work section.
>
> Discuss dataset size, class imbalance, label harmonization, domain shift, possible residual group leakage among unmatched images, the limited number of seeds, and the exploratory nature of the external evaluation.
>
> Suggest realistic future improvements, such as larger field datasets, additional random seeds, stronger domain adaptation methods, and further refinement ablations.
>
> Distinguish limitations that were directly observed from risks that could not be fully ruled out.

### Prompt 24 — Figures and Scientific Visualization

> Recommend and prepare a compact set of scientific figures that communicate the main findings clearly.
>
> Include comparative Macro F1 plots for PlantVillage and PlantDoc, a domain-drop comparison, and a paired bootstrap confidence-interval plot.
>
> Ensure that every figure uses consistent model names, readable labels, appropriate legends, and publication-quality resolution.
>
> Avoid decorative graphics that do not contribute to the research argument.

### Prompt 25 — Final Academic Report

> Prepare a complete LaTeX research report for the project "Few-Shot Plant Disease Recognition Using Self-Supervised Vision Foundation Models."
>
> Structure the report as a formal academic document containing an abstract, introduction, related work, methodology, experimental setup, results, discussion, limitations, conclusion, and references.
>
> Include the actual experimental results and clearly explain the dataset preparation, group-aware splitting, frozen DINOv2 feature extraction, few-shot support sets, classifier configurations, prototype refinement, and statistical analysis.
>
> Use professional academic English, consistent terminology, properly formatted tables, and figure placeholders with explicit filenames.
>
> Do not invent experimental results, citations, or implementation details. Clearly mark any personal or institutional information that must be completed by the student.

### Prompt 26 — Reproducibility and Final Quality Review

> Conduct a final scientific and technical review of the report and project deliverables.
>
> Check consistency between the reported metrics, tables, figures, experimental configurations, and conclusions.
>
> Verify that the report does not claim that prototype refinement is universally superior or statistically significant when the results do not establish this.
>
> Check that the limitations of the PlantDoc external test set and the unperformed ablations are explicitly documented.
>
> Review the LaTeX structure, references, figure paths, and reproducibility instructions.
>
> Produce a final checklist of items that must be verified before submission.
