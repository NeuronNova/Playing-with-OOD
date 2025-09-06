# NOREGRET-CNN : *A digit classifier that knows when to stay silent.*




## Overview
Most digit classifiers will confidently assign a label - even when the input is pure noise, a letter, or just a random sketch. This project aims to go beyond that . It aims to build a classifier that not only is capable of classifying inputs but would also be able to classify OOD inputs correctly.

Checkout the streamlit application here :  https://noregret-cnn.streamlit.app/

It’s a digit classifier that refuses to guess when it's uncertain.  
Unlike traditional models, it includes a rejection mechanism that avoids overconfident misclassification of out-of-distribution inputs.

Built on MNIST, NOREGRET uses calibrated confidence and embedding-based reasoning to reject inputs that don't resemble any known digit.

This project combines:
- A lightweight **Convolutional Neural Network (CNN)**  
- **Embedding-based** prototype comparison  
- **Beta-distribution-based** confidence modeling  
- A simple Streamlit interface for drawing digits and testing

---

## OOD Rejection Comparison

Out-of-distribution (OOD) detection is a core feature of NOREGRET-CNN.  
We evaluated two strategies on **EMNIST Letters (A-Z)**:

| Method                   | Rejection Rate | Visualization | Notes |
|---------------------------|----------------|---------------|---------------|
| KNN-based distance              | 79.6%          | ████████████████████████░░░░ (79.6%) |Rejects samples based only on distance to nearest neighbors in the training set. Simple and intuitive, but sensitive to normalization and thresholds.
| Prototype-based Rejection | 83.0%          | █████████████████████████░░░ (83.0%) |Rejects samples if **distance to class prototype** is high **or** **beta-calibrated confidence** is low. Adds principled uncertainty modeling while retaining interpretability

> ⚡ **Key Takeaways:**  
> - Both methods leverage CNN embeddings to detect unfamiliar inputs.  
> - **KNN** is a naive distance-based approach — works reasonably but lacks explicit uncertainty modeling.  
> - **Prototype-based rejection** builds on the same idea but introduces **class prototypes and beta-calibrated confidence**, making the rejection decision **robust, principled, and interpretable**.  
> - The slightly higher rejection rate (83% vs 79.6%) demonstrates improved coverage of OOD inputs.  

---
##  What Makes It Different

CNN trained on MNIST  
1. **Beta Calibration** to reduce overconfidence  
2. **Intermediate embeddings** extracted from the model  
3. **Cosine similarity** to class prototypes  
4. Rejects inputs that are far from any known digit cluster or have low confidence

---

## 📷 Example Behavior

When the given input is a digit : 
<img width="1474" height="878" alt="image" src="https://github.com/user-attachments/assets/f8fde0c0-b106-4621-9216-f0c413df7aa6" />

When we test the model with OOD : 
<img width="1498" height="869" alt="image" src="https://github.com/user-attachments/assets/08dd6330-568b-4b2f-8f83-44d5a47d548a" />


Even a small model can learn when to say,  
**"This doesn’t look like any digit I know."**

---

## Use Cases

- Out-of-distribution (OOD) rejection  
- Safe AI deployments in constrained domains  
- Teaching calibration and interpretability concepts  
- Baseline for open-set recognition with minimal compute

---

## Tech Stack

- Tensorflow  
- Scikit-learn  
- NumPy  
- Matplotlib

---

## ⚙️ How It Works

1. Train a simple CNN on MNIST  
2. Apply **Beta Calibration** to output probabilities  
3. Extract embedding from the penultimate layer  
4. Compute **cosine similarity** to pre-computed **class prototypes** (centroids)  
5. If similarity is low **or** confidence is below a threshold → **reject**

---

## 💡 Future Directions

- Generalize to other datasets (letters, symbols, multi-modal inputs)  
- Apply to real-world digit inputs (handwritten forms, tablets)  
- Extend to multi-class classifiers beyond digits  
- Visualize embedding space in real time

---

