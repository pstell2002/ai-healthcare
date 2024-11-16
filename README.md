# Melanoma Detection Using HAM10000 and Kaggle Datasets

This repository contains two Jupyter notebooks designed to preprocess images and train convolutional neural network (CNN) models to detect melanoma from skin lesion images. The two notebooks focus on different datasets: one uses the **HAM10000 dataset**, while the other uses a **balanced Kaggle dataset**. Both notebooks implement hair removal preprocessing techniques and train models using TensorFlow/Keras. However, there are key differences in how the data is handled, balanced, and evaluated.

## Notebooks Overview

### 1. `ham1000_ISIC_dataset_hairless.ipynb`
This notebook focuses on the **HAM10000 dataset**, which is a well-known dataset for skin lesion classification. The key characteristics of this notebook include:
- **Dataset**: The HAM10000 dataset contains around 10,000 dermatoscopic images of different types of skin lesions, including benign and malignant cases.
- > Link to dataset: https://drive.google.com/drive/u/0/folders/1LQ-qr9lQ-fUZF4jRwysel_R9UfH6su5B
  > Note: currently the notebook assumes you are using the `ISIC-images-preprocessed` folder which contains the images after masking the lesion and inpainting the hair.
- **Class Imbalance**: The HAM10000 dataset is highly imbalanced, with a significant majority of benign cases (about 85%) compared to malignant cases (about 15%). This imbalance requires special handling during training.
- **Preprocessing**: Hair removal preprocessing is applied to each image using OpenCV's morphological operations to remove hair artifacts that could interfere with model performance.
- **Model Training**: The model is trained using class weights to give more importance to malignant samples due to the class imbalance.
- **Evaluation Metrics**: The model is evaluated using accuracy, precision, recall, F1-score, specificity, sensitivity, and AUC (Area Under Curve). Given the imbalance in the dataset, metrics like recall (sensitivity) for malignant cases are particularly important.
- **Confusion Matrix & ROC Curve**: A confusion matrix is plotted to visualize how well the model distinguishes between benign and malignant cases. An ROC curve is plotted to show model performance across different thresholds.

### 2. `ham1000_kaggle_balanced_dataset_hairless.ipynb`
This notebook uses a **balanced dataset from Kaggle**, which has been curated to contain an equal number of benign and malignant samples. The key characteristics of this notebook include:
- **Dataset**: The Kaggle dataset used here has been balanced to contain an equal proportion of benign and malignant cases. This balancing helps mitigate issues related to class imbalance during training.
- > Link to dataset: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images/data 
- **Class Balance**: Unlike the HAM10000 dataset, this dataset does not require class weighting during training because it already contains an equal number of samples from each class.
- **Preprocessing**: Similar hair removal preprocessing techniques are applied as in the first notebook.
- **Model Training**: Since the dataset is balanced, no class weights are used during training. This allows for more straightforward training without needing to adjust for class imbalance.
- **Evaluation Metrics**: Similar metrics are used for evaluation (accuracy, precision, recall, F1-score, specificity, sensitivity, AUC). However, since the dataset is balanced, metrics like accuracy may be more reliable indicators of overall performance compared to the imbalanced case in the first notebook.
- **Confusion Matrix & ROC Curve**: A confusion matrix and ROC curve are also plotted in this notebook. Since the dataset is balanced, you can expect more evenly distributed true positives and true negatives in the confusion matrix.

## Key Differences Between Notebooks

### 1. Dataset Imbalance
- **HAM10000 Notebook (`ham1000_ISIC_dataset_hairless.ipynb`)**:
  - The dataset is highly imbalanced with approximately 85% benign cases and 15% malignant cases.
  - Class weighting is applied during training to handle this imbalance.
  - Evaluation metrics like recall (sensitivity) for malignant cases are critical due to the imbalance.

- **Kaggle Balanced Dataset Notebook (`ham1000_kaggle_balanced_dataset_hairless.ipynb`)**:
  - The dataset has been balanced with an equal number of benign and malignant samples.
  - No class weighting is needed during training because both classes are equally represented.
  - Accuracy becomes a more reliable metric because there’s no imbalance.

### 2. Model Training Strategy
- In the **HAM10000 notebook**, class weights are used during training due to class imbalance. This ensures that malignant cases have more influence on model updates despite being underrepresented in the data.
- In contrast, in the **Kaggle balanced notebook**, no class weights are necessary since both classes are equally represented. This simplifies training but may not reflect real-world scenarios where melanoma detection often involves imbalanced datasets.

### 3. Evaluation Focus
- In the HAM10000 notebook:
  - Metrics like recall (sensitivity) for detecting malignant cases are emphasized because it’s crucial not to miss malignant diagnoses in medical applications.
  - Specificity (true negative rate) also plays an important role but may be less critical than sensitivity due to the life-threatening nature of melanoma.

- In the Kaggle balanced notebook:
  - Metrics like accuracy may be more reliable since both classes are equally represented.
  - Precision and recall are still important but may not require as much adjustment as in an imbalanced setting.

## Key Features

### Hair Removal Preprocessing
Both notebooks include a preprocessing step that removes hair artifacts from skin lesion images using OpenCV’s morphological operations (comparison before and after hair removal is available on the Kaggle dataset in `ham1000_kaggle_balanced_dataset_hairless.ipynb`:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
_, hair_mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
hairless_image = cv2.inpaint(image, hair_mask, 6, cv2.INPAINT_TELEA)
```

### Model Architecture
The CNN architecture includes multiple convolutional layers followed by max-pooling and dropout layers:

```python
model = Sequential([
    Input(shape=(256, 256, 3)),
    Augmentation,
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),
    Flatten(),
    Dense(units=128, activation="relu"),
    Dropout(0.5),
    Dense(units=2, activation="softmax"),
])
```

### Evaluation Metrics
Both notebooks evaluate models using several metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- Sensitivity
- AUC (Area Under Curve)

The confusion matrix visualizes how well each model distinguishes between benign and malignant cases.

## Requirements

To run these notebooks locally or on a cloud platform like Google Colab or Kaggle Notebooks:
- Python 3.x
- TensorFlow/Keras
- OpenCV (`cv2`)
- Matplotlib for plotting
- Seaborn for visualizing confusion matrix
- Scikit-learn for metrics like precision/recall and computing class weights

You can install the required libraries using:

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn tqdm pandas numpy
```

3. Open either of the notebooks in Jupyter or Google Colab:
   - `ham1000_ISIC_dataset_hairless.ipynb`
   - `ham1000_kaggle_balanced_dataset_hairless.ipynb`

4. Run all cells in sequence.

## Results

After training the model on either dataset:
- The final accuracy on the test set will be displayed.
- The confusion matrix will show how well the model distinguishes between benign and malignant cases.
- The ROC curve will be plotted to show how well the model performs across different thresholds.
