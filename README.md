# Fabric Defect Detection

## Project Overview

Fabric Defect Detection is an AI-based image processing project designed to classify fabric images as either **Stain** or **Defect Free**.

The project focuses on applying image processing techniques to fabric images before using a machine learning model for classification.

## Objective

The main objective is to demonstrate how image processing and machine learning can be combined for basic textile defect detection.

The system:

* Takes a fabric image as input.
* Applies multiple image processing techniques.
* Extracts image pixel features.
* Uses a Decision Tree Classifier.
* Predicts whether the fabric contains a stain or is defect-free.
* Visually displays the different stages of image processing.

## Dataset

The project uses a Fabric Stain Dataset containing **466 images**.

| Category    | Number of Images |
| ----------- | ---------------: |
| Stain       |              398 |
| Defect Free |               68 |
| **Total**   |          **466** |

The dataset contains fabric images with stain types including ink, dirt, and oil stains.

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Streamlit

## Image Processing Pipeline

The uploaded fabric image passes through the following stages:

### 1. Image Resizing

The image is resized to **64 × 64 pixels** to provide a consistent input size.

### 2. Grayscale Conversion

The color image is converted into grayscale to simplify image analysis and reduce the number of pixel values.

### 3. Median Filtering

A median filter is applied to reduce small amounts of noise while preserving important image details.

### 4. Histogram Equalization

Histogram equalization improves image contrast and makes intensity differences more visible.

### 5. Edge Detection

Canny Edge Detection is used to identify important edges and boundaries in the processed image.

### 6. Image Segmentation

Otsu's thresholding method separates different intensity regions in the image.

### 7. Defect Highlight

The segmentation result is used to create a visual highlight of potential defect regions.

This provides a Before/After comparison to make the image-processing result easier to observe.

## Machine Learning

A **Decision Tree Classifier** is used for final classification.

The processed grayscale image is converted into numerical pixel features and supplied to the Decision Tree.

The model predicts one of two classes:

* **0 → Defect Free**
* **1 → Stain**

The dataset is divided into:

* **80% training data**
* **20% testing data**

Class balancing is used during Decision Tree training because the dataset contains considerably more stain images than defect-free images.

## Model Performance

The model achieved a test accuracy of:

**72.34%**

### Confusion Matrix

```text
[[ 1 13]
 [13 67]]
```

The results show that the model performs better on the Stain class than the Defect Free class.

The difference is partly related to the imbalance in the dataset, where there are 398 stain images compared with only 68 defect-free images.

## Streamlit Application

The project includes an interactive Streamlit application.

The user can upload a JPG, JPEG, or PNG fabric image.

The application displays:

1. Original image
2. Grayscale image
3. Noise-removed image
4. Contrast-enhanced image
5. Edge-detected image
6. Segmented image
7. Before/After defect-highlight comparison
8. AI prediction
9. Image brightness
10. Detected edge pixels
11. Dataset and model information

## Project Workflow

```text
Fabric Image
      ↓
Image Resizing
      ↓
Grayscale Conversion
      ↓
Median Filtering
      ↓
Histogram Equalization
      ↓
Edge Detection
      ↓
Image Segmentation
      ↓
Defect Highlight
      ↓
Feature Extraction
      ↓
Decision Tree Classifier
      ↓
Stain / Defect Free
```

## Project Structure

```text
Fabric-Defect-Detection/
│
├── dataset/
│   └── archive (3)/
│       ├── annotations/
│       └── images/
│           ├── defect_free/
│           └── stain/
│
├── models/
│   └── decision_tree.pkl
│
├── src/
│   └── train_model.py
│
├── outputs/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
streamlit run app.py
```

### 3. Upload an Image

Upload a fabric image through the application to view the image-processing stages and obtain the model's classification.

## Limitations

* The dataset is relatively small.
* The dataset is imbalanced between the two classes.
* The current model has lower performance for the Defect Free class.
* The defect highlight is a visualization based on image segmentation and should not be interpreted as an exact defect boundary.
* The system is intended as an academic demonstration and not as an industrial quality-control system.

## Conclusion

This project demonstrates the use of image processing techniques and machine learning for fabric defect classification.

The combination of preprocessing, enhancement, edge detection, segmentation, visualization, and Decision Tree classification provides a complete workflow for studying fabric defect detection using AI and image processing.
