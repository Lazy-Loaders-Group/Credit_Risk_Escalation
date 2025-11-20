## Project Proposal

Consider an application area where safety is critical. E.g., medical diagnostics or autonomous vehicles. When using predictive models in such areas just giving a prediction is not sufficient. In addition to accurate predictions, the model should be well calibrated and provide uncertainty estimates for the predictions it gives. Furthermore, when there is high uncertainty the model could abstain from giving a prediction and refer to a human expert (ML with rejection).

Pick a safety critical application area, where ML-based prediction models have been applied but their uncertainty estimation is not explored sufficiently.

For the area you picked provide a short report (about 3-6 pages) with the following sections.

Background and introduction to the area
Literature review including about 5 key papers (this may include some attempts to estimate model uncertainty)
Problem statement: State why existing work doesn't address prediction uncertainty adequately.
Proposed method: propose an uncertainty estimation technique that you can use (you can change this later as needed as you progress). You may refer to this document for some useful information.
List of available datasets you can use.

## Project Report

Implement your proposed methodology and perform experiments to evaluate it.

Please upload your Project report as a PDF file here. The report must contain the following.

Introduction, Literature review and Method (you can simply copy these sections from your proposal and update if there's any change)
Details of the experiments carried out and the data used
Results
Conclusions

==>> What we discussed about the project report content

Required Sections:

1. Introduction, Literature Review, and Method


    - Copy from proposal, update if methodology changed
    - Your proposal already has these well-documented

2. Details of Experiments and Data Used


    - Dataset: Lending Club (which you used)
    - Data preprocessing steps
    - Train/validation/test splits
    - Feature engineering details

3. Results


    - Baseline model performance (AUC: 0.823)
    - Uncertainty quantification validation (0.324 correlation)
    - Escalation system metrics:
        - Automation rate: 78.3%
      - Automated accuracy: 88.76%
      - Cost savings: 20.9%
    - Visualizations (calibration plots, uncertainty distributions, SHAP)

4. Conclusions


    - Summary of achievements
    - Comparison to baseline (no uncertainty)
    - Limitations and future work

---

Discussion Points:

What needs updating from the proposal?

- Method: You used Bootstrap Ensemble (30 models) + Platt scaling (not MC Dropout/Temperature scaling as originally proposed)
- Dataset: Lending Club (~210K samples after preprocessing)

What to emphasize in Results?

- Your actual metrics exceed the targets
- Cost-benefit analysis shows clear business value
- SHAP explainability for regulatory compliance
