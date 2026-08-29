# Patho-XAG

Patho-XAG is our research project for colorectal histopathology image analysis. 
The project is initially developed as part of our capstone study focusing on colorectal adenoma classification using the CAMEL dataset.

## Current Research Scope

At the current stage, we focus on:

- Colorectal adenoma vs. normal classification
- Whole-image vs. patch-based image representation
- Multiple Instance Learning (MIL)
- Comparison of MIL aggregation methods
- Comparison of histopathology feature extractors

## Dataset
We use the CAMEL dataset for the project.

This is the colorectal adenoma dataset for "CAMEL: A Weakly Supervised Learning Framework for Histopathology Image Segmentation" (ICCV 2019). [Paper](http://arxiv.org/abs/1908.10555).

The dataset is hosted on [Google Drive](https://drive.google.com/drive/folders/1brr8CnU6ddzAYT157wkdXjbSzoiIDF9y) or [Baidu Drive](https://pan.baidu.com/s/1kk3rUgFkY7b3FX9g--w_5g) with password ```x2o5```

## Research Questions
We proposed 3 following research questions:

### RQ1 — Input Representation

Should colorectal histopathology ROIs be processed as whole images or divided into smaller patches?

### RQ2 — Patch Aggregation

Which aggregation method best combines information from multiple patches for ROI-level classification?

### RQ3 — Feature Extraction

Which feature extractor provides the most effective representation for colorectal adenoma classification?

## Project Structure

```text
Patho-XAG/
├── configs/
├── notebooks/
├── scripts/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── outputs/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt