# Med-Vision
### A Deep Learning Approach to Multi-Class Disease Classification using Computer Vision

## **Overview**
The **Med-Vision** project is a deep learning-based computer vision model designed for **multi-class disease classification**. The model is capable of identifying and categorizing diseases such as **Acne, Conjunctivitis, Folliculitis, Hives Urticaria, Nail Fungus and Cracked Nails, Ringworms (Skin Fungal Infection), Stye (Inflammation around eyes), and Warts** 

A custom dataset of **4,208 images** was initially used, but after applying **data augmentation techniques**, the dataset expanded to **12,000+ images**, ensuring a more robust and generalizable model.
The dataset is uploaded on the Google Drive here, [Med-Vision-Dataset](https://drive.google.com/drive/u/1/folders/17tyChnUsX3WztwtLIKStYSOR6t0lsLVT), it is because of its huge size, pushing it to Github was troublesome.

The primary goal of this project was to build an efficient classification model leveraging **transfer learning** techniques with **CNN architectures**, ultimately achieving a high classification accuracy of **~90%**.

The workflow includes:
- **Data collection and augmentation**
- **Preprocessing and feature extraction**
- **Model selection, training and Testing Functions**
- **Performance evaluation and Predictions**

This project documents the challenges encountered, the methodologies adopted, and the rationale behind choosing specific models for final deployment.

---

## **Project Analysis**

### **Dataset and Preprocessing**
- The dataset initially contained **4,208 images**, covering 8 different diseases as classes.
- Applied **image augmentation** techniques such as rotation, flipping, zooming, and brightness adjustments to increase the dataset to **12,000 images**.
- Standardized images to a fixed size before feeding them into the model.
- Used **PIL (Pillow) and Torchvision** to handle image transformations.

### **Modeling Approach**
- Used **transfer learning** with pre-trained CNN architectures to enhance accuracy and reduce training time.
- Two state-of-the-art models were trained and evaluated:
  - **EfficientNet_B0**
- The model was then fine-tuned using **Adam optimizer** and **cross-entropy loss**.

### **EfficientNet_B0 Model**
- **Objective**: Achieve high accuracy with a lightweight, efficient architecture.
- **Performance**:
  - **Test Accuracy**: **0.85 (85%)**
- **Strengths**:
  - Optimized for computational efficiency.
  - Provides high accuracy with fewer parameters.
- **Challenge faced by Me**:
  - Required very long time to get trained for fine-tuning for just few number of Epochs.

### **Libraries Used**
- `Python`: Primary programming language for implementation.
- `PyTorch, Torchvision`: Deep learning framework and pre-trained model access.
- `Pillow`: Image loading and manipulation.
- `NumPy, Pandas`: Data handling and preprocessing.
- `Matplotlib`: Visualization of model performance.

---

## **Results Summary**

| **Model**            | **Test Accuracy** |
|---------------------|-----------------|
| EfficientNet_B0     | **0.90 (90%)**  |

---

## **Performance Visualization**
The model's training process is illustrated with the following loss curves:

**Loss Curve and Accuracy Curve for both train and test dataset**
**EfficientNet_B0 Loss Curve**
   ![EfficientNet Loss Curve](Loss_and_Accuracy_Curves_(Images)/For_10_Epochs.png)


These loss curves provide insights into the training convergence, showing how the models improved over epochs.

---

## **Conclusion**
The **Med-Vision** model successfully classifies 8 diseases' classes with high accuracy, leveraging transfer learning. EfficientNet_B0 is the model used for this process, as it outperforms because it is not really expensive to use unlike ResNet, ShuffleNet etc., making it the preferred model for deployment. Future improvements could involve:
- Collecting more diverse data to enhance generalization.
- Fine-tuning hyperparameters to boost accuracy.
- Deploying the model as a web or mobile application for real-world use cases.

### **This Project was built by Vidit Saini**
