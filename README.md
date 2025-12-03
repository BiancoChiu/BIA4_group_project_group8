# Droso Wing Classifier
### Gene + Sex Multi-Model Fusion Classifier (CNN + Handcrafted Features) with Batch & Gradio Inference

This repository provides a complete pipeline for *Drosophila wing image classification*, supporting:

- ✔ **Gene classification model**
- ✔ **Sex classification model**
- ✔ **CNN + 74-dim handcrafted feature fusion**
- ✔ **Batch inference (multi-model)**
- ✔ **Interactive Gradio Web App**
- ✔ **Reproducible feature extraction pipeline**

All paths use **relative paths**, so the project can run anywhere after cloning.

## Project Structure

```
Droso/
│── batch_inference_v2.py
│── gradio_app_gene_sex.py
│
│── feature_extraction/
│     └── full_feature_extractor_v3.py
│
│── models/
│     ├── fusion_cnn_mlp_model/
│     │      ├── fusion_best.pth
│     │      └── label2idx.json
│     └── fusion_cnn_mlp_sex_model_try_best/
│            ├── fusion_best.pth
│            └── label2idx.json
│
│── data/
│     ├── reference_feature.csv
│     └── crop_testing/
│
└── output/
      └── batch_inference_gene_sex.csv
```

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n droso python=3.10
conda activate droso
pip install -r requirements.txt
```

## Batch Inference (Gene + Sex)

Run:

```bash
python batch_inference_v2.py --img_dir ./data/crop_testing --out_csv ./output/batch_inference_gene_sex.csv
```

Custom:

```bash
python batch_inference_gene_sex.py --img_dir ./data/40X_sampled_png --out_csv ./output/my_results.csv
```

## Gradio Web Application

Run:

```bash
python gradio_app_gene_sex.py
```

Visit: `http://127.0.0.1:7860`

Features:

- Upload one wing image
- Extract handcrafted features
- Predict gene + sex
- Show full probability distribution

## Notes

- All paths use **relative paths**


