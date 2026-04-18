# 🚀 E-commerce Return Prediction (Real-World ML)

## 📌 Problem Statement

In e-commerce, predicting whether an order will be returned or cancelled is crucial for reducing losses and improving logistics.

This project builds a **machine learning model to predict return risk at the time of purchase**, using only information available before delivery.

---

## 📊 Dataset

* **Source:** Brazilian E-commerce Dataset (Olist)
* ~117K orders after merging multiple tables
* Highly imbalanced:

  * Returned: ~0.47%
  * Not Returned: ~99.5%

---

## ⚙️ Approach

### 🔹 Data Processing

* Merged datasets: orders, customers, payments, products
* Removed duplicates and cleaned missing values

### 🔹 Feature Engineering (Leakage-Free)

Only features available at purchase time:

* `purchase_hour`
* `purchase_day`
* `total_price`
* `payment_value`
* `payment_installments`

⚠️ Removed `delivery_delay` to avoid data leakage

---

## 🤖 Model

* Algorithm: **XGBoost Classifier**
* Handled class imbalance using `scale_pos_weight`
* Used **Stratified K-Fold Cross Validation**

---

## 📈 Results (Realistic)

* **ROC-AUC:** ~0.48–0.49
* **Accuracy:** ~91–98% (misleading due to imbalance)
* **Precision (Returns):** ~0.01
* **Recall (Returns):** ~0.03–0.09

---

## 🔍 Key Observations

### ❗ 1. Severe Class Imbalance

* Model struggles to learn rare return patterns
* Accuracy is not a reliable metric here

### ❗ 2. Weak Predictive Signal

* Available features have **low correlation with returns**
* Model performs close to random guessing (ROC-AUC ≈ 0.5)

### ❗ 3. Feature Importance

* Top feature: `payment_value`
* No single feature strongly drives prediction

---

## ⚠️ Critical Insight (Most Important)

> When using `delivery_delay`, model achieved **ROC-AUC ~0.98**

BUT:

* This is **data leakage**
* It uses post-delivery information
* Not available at prediction time

---

## 🧠 Key Learnings

* High accuracy can be misleading in imbalanced datasets
* Feature quality > model complexity
* Data leakage can artificially inflate performance
* Real-world ML requires strict feature validation

---

## 🛠️ Tech Stack

* Python
* Pandas
* XGBoost
* Scikit-learn
* Matplotlib / Seaborn

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
kaggle datasets download -d olistbr/brazilian-ecommerce
unzip brazilian-ecommerce.zip
```

Then run the notebook.

---

## 📂 Project Structure

```
ecommerce-return-prediction/
│── ecommerce_return_prediction.ipynb
│── README.md
│── requirements.txt
│── .gitignore
```


## ⭐ Final Note

This project demonstrates an important real-world lesson:

> A model is only as good as the data and features it is built on.

Removing leakage revealed that predicting returns at purchase time is a **challenging problem with weak signal**, highlighting the need for better features or additional data sources.
