#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[6]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
columns = ["Class", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]

data = pd.read_csv(url, names=columns)

# Encode target labels: L=0, B=1, R=2
data['Class'] = data['Class'].map({'L': 0, 'B': 1, 'R': 2})


# In[7]:


X = data.drop("Class", axis=1)  # Features
y = data["Class"]  


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[11]:


model = LogisticRegression(
   # multi_class="multinomial",   (no requirement in new scikit learn)
    solver="lbfgs",
    max_iter=500,
    class_weight="balanced",  # handle imbalance
    random_state=42  
)
model.fit(X_train, y_train)


# In[12]:


y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# In[15]:


print("\n--- Predict Balance Scale Outcome ---")
try:
    left_weight = int(input("Enter Left-Weight (1-5): "))
    left_distance = int(input("Enter Left-Distance (1-5): "))
    right_weight = int(input("Enter Right-Weight (1-5): "))
    right_distance = int(input("Enter Right-Distance (1-5): "))

    new_input = [[left_weight, left_distance, right_weight, right_distance]]
    predicted_class = model.predict(new_input)[0]

    # Map numeric class back to label
    class_mapping = {0: "L", 1: "B", 2: "R"}
    print("Predicted class for your input:", class_mapping[predicted_class])

except ValueError:
    print("Please enter valid integer values between 1 and 5.")


# In[ ]:




