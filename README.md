# Detection: Flooding by Images and Metadata

[![Task Preview](https://multimediaeval.github.io/2017-Multimedia-Satellite-Task/Preview_DIRSM.png)](https://multimediaeval.github.io/2017-Multimedia-Satellite-Task/)

## 📖 Introduction
This project implements a multi-modal deep learning approach to identify flooding events from social media streams. By fine-tuning **Vision Transformer (ViT)** and **DeBERTa** on a flood image dataset, the model effectively combines visual and textual features, achieving significantly higher accuracy than single-modality baselines.

## 🎯 Task Description
The goal of this task is to retrieve all images which show direct evidence of a flooding event from social media streams, independently of a particular event. The objective is to design a system/algorithm that, given any collection of multimedia images and their metadata (e.g., YFCC100M, Twitter, Wikipedia, news articles), is able to identify those images related to a flooding event.

*Note that only images conveying direct evidence of a flooding event are considered True Positives. Specifically, we define images showing "unexpected high water levels in industrial, residential, commercial, and agricultural areas" as providing evidence.*

**Main Challenges:**
- Proper discrimination of water levels in different areas (e.g., distinguishing a normal lake vs. high water on a street).
- Consideration of various types of flooding events (e.g., coastal, river, pluvial flooding).

## 🧠 Method
The architecture combines the power of two robust models:
- **Vision Transformer (ViT):** Extracts complex visual features from images.
- **DeBERTa-v3-small:** Analyzes and understands the contextual semantics of image descriptions and metadata.

Applying multi-modal deep learning to fuse these visual and textual features leads to a comprehensive understanding, resulting in better accuracy than relying on a single modality.

## 📊 Dataset
The dataset is structured on Kaggle as follows:
![Link](https://www.kaggle.com/datasets/phmngcbch/2025-sum-dpl-302-rn)
```text
/kaggle/input/2025-sum-dpl-302-m/
│
├── devset_images/                       
│   └── devset_images/ (5,280 images)
├── devset_images_metadata.json (Metadata)      
├── devset_images_gt.csv (Ground truth labels)           
├── testset_images/                   
│   └── testset_images/ (1,320 images)
└── test.csv (Test metadata: image_id, title, description, user_tags)
```


### Result
- Achieved 0.91 private and public scores on the flood image classification task using mAP@250

![Samplle Data/Screenshot%2025-08-05%142821.png](https://github.com/bachPN73/Prediction-Flooding-by-Images-and-Metadata/blob/main/Samplle%20Data/Screenshot%202025-08-05%20153834.png)

### 🚀 Quick Setup & Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/bachPN73/Detection_Flooding_by_images_and_metadata.git](https://github.com/bachPN73/Detection_Flooding_by_images_and_metadata.git)
cd Detection_Flooding_by_images_and_metadata
```
**2. Install dependencies:
```bash
pip install torch torchvision transformers scikit-learn pandas numpy Pillow
```
