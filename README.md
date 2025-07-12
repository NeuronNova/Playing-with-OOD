# NOREGRET  
*A digit classifier that knows when to stay silent.*

---

## 🧠 Overview

Most digit classifiers will confidently assign a label , even when the input is pure noise, a letter, or just a random sketch.

**NOREGRET** is different.  
It’s a digit classifier that refuses to guess when it's uncertain.

Built on MNIST, NOREGRET uses calibrated confidence and embedding-based reasoning to reject inputs that don't resemble any known digit.

---

##  What Makes It Different

✅ CNN trained on MNIST  
1. **Beta Calibration** to reduce overconfidence  
2. **Intermediate embeddings** extracted from the model  
3. **Cosine similarity** to class prototypes  
4. Rejects inputs that are far from any known digit cluster or have low confidence

The goal: don't force predictions — **know when not to classify**.

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

- PyTorch  
- Scikit-learn  
- NumPy  
- Matplotlib

---

## 📁 Project Structure

