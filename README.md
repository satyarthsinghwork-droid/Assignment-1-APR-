# Logistic Regression on Balance Scale Dataset

This project applies **multiclass logistic regression** to the classic **Balance Scale dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/balance+scale).  
It demonstrates dataset loading, model training, evaluation, and interactive prediction using user input.

---

## 📊 Dataset Information
- **Source:** UCI Machine Learning Repository  
- **URL used in script:**  
  ```
  https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data
  ```

- **Features:**
  - Left-Weight (1–5)  
  - Left-Distance (1–5)  
  - Right-Weight (1–5)  
  - Right-Distance (1–5)  

- **Target Classes:**
  - `L` → Balance tips to the left → encoded as `0`  
  - `B` → Balanced → encoded as `1`  
  - `R` → Balance tips to the right → encoded as `2`  

- **Dataset Size:** 625 samples  
- **Imbalance:** `B` (Balanced) occurs in only ~8% of cases.

---

## ⚙️ Model Details
- Algorithm: **Logistic Regression**  
- Library: `scikit-learn`  
- Parameters:
  ```python
  LogisticRegression(
      solver="lbfgs",
      max_iter=500,
      class_weight="balanced",  # handles class imbalance
      random_state=42
  )
  ```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install pandas scikit-learn
```

### 2. Run the Script
```bash
python balance_logistic.py
```

### 3. What Happens
- Loads dataset from the UCI link  
- Splits into train/test sets  
- Trains logistic regression model  
- Prints:
  - Confusion Matrix  
  - Classification Report  
  - Accuracy Score  
- Prompts for **user input** to make predictions

---

## 🎮 Example Usage

### Model Evaluation Output
```
Confusion Matrix:
 [[81  6  0]
 [ 1 14  0]
 [ 0  4 82]]

Classification Report:
               precision    recall  f1-score   support
           0       0.99      0.93      0.96        87
           1       0.58      0.93      0.72        15
           2       1.00      0.95      0.98        86

Accuracy Score: 0.94
```

### User Prediction
```
--- Predict Balance Scale Outcome ---
Enter Left-Weight (1-5): 3
Enter Left-Distance (1-5): 2
Enter Right-Weight (1-5): 4
Enter Right-Distance (1-5): 1

Predicted class for your input: R
```

---

## 📌 Notes
- The dataset is **imbalanced**; most examples are `L` or `R`.  
- `class_weight="balanced"` improves fairness across classes.  
- The **Balanced (B)** class remains harder to predict.  
- Possible improvements:
  - Oversampling (e.g., SMOTE)  
  - Using tree-based models (Random Forest, XGBoost)  

---

## 📚 References
- Dataset: [UCI Balance Scale](https://archive.ics.uci.edu/ml/datasets/balance+scale)  
- Paper: Siegler, R. S. (1976). *Three Aspects of Cognitive Development*. Cognitive Psychology, 8, 481–520.
