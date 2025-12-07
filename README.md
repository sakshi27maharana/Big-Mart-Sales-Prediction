---

# **Big Mart Sales Prediction**

This repository contains an end-to-end machine learning project for predicting product-level sales across BigMart outlets. The goal is to build a predictive model using historical sales data from 2013 and understand the properties of products and outlets that influence sales performance.

---

## 📌 **Project Overview**

BigMart has collected sales data for **1559 products** sold across **10 stores**. The dataset includes product attributes, store metadata, and historical sales values.
Our objective is to:

* Explore, clean, and preprocess the dataset
* Handle missing values and inconsistent entries
* Engineer meaningful features
* Build and evaluate multiple ML models
* Predict `Item_Outlet_Sales` for the test dataset
* Generate an output file following the given submission format

The project's final evaluation metric is **Root Mean Square Error (RMSE)**.

---

## 📂 **Dataset Description**

The data consists of two CSV files:

### **Train Dataset (8523 rows)**

Contains input features along with the target column `Item_Outlet_Sales`.

### **Test Dataset (5681 rows)**

Contains the same feature set *without* the sales values to be predicted.

---

## 🧾 **Data Dictionary**

### **Common Features**

| Variable                    | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `Item_Identifier`           | Unique product ID                              |
| `Item_Weight`               | Weight of the product                          |
| `Item_Fat_Content`          | Whether the product is low-fat or regular      |
| `Item_Visibility`           | Percentage of shelf area allocated to the item |
| `Item_Type`                 | Category of the product                        |
| `Item_MRP`                  | Maximum Retail Price                           |
| `Outlet_Identifier`         | Unique store ID                                |
| `Outlet_Establishment_Year` | Year the store was established                 |
| `Outlet_Size`               | Physical size of the store                     |
| `Outlet_Location_Type`      | Tier of the city                               |
| `Outlet_Type`               | Type of outlet (grocery store or supermarket)  |

### **Target Variable (Train Only)**

| Variable            | Description                         |
| ------------------- | ----------------------------------- |
| `Item_Outlet_Sales` | Sales of the product in that outlet |

---

## 🧮 **Submission Format**

Your final output must contain:

| Column              | Description           |
| ------------------- | --------------------- |
| `Item_Identifier`   | Unique product ID     |
| `Outlet_Identifier` | Unique store ID       |
| `Item_Outlet_Sales` | Predicted sales value |

---

## 🧪 **Evaluation Metric**

Performance is measured using **Root Mean Square Error (RMSE)** between your predictions and the true sales values.

---

## 🔍 **Public & Private Leaderboards**

* **Public Score:** Based on 25% of test data
* **Private Score:** Based on remaining 75% (revealed after competition ends)

---

## 🛠️ **Project Workflow**

1. **Exploratory Data Analysis (EDA)**
2. **Missing Value Imputation**
3. **Feature Engineering**
4. **Encoding Categorical Variables**
5. **Model Training (Linear Regression, Random Forest, XGBoost, etc.)**
6. **Hyperparameter Optimization**
7. **Model Evaluation (RMSE)**
8. **Final Prediction on Test Data**
9. **Submission File Generation**

---

## 📁 **Repository Structure**

```
Big-Mart-Sales-Prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── big_mart_analysis.py
│
├── submission/
│   └── final_submission.csv
│
├── README.md
└── requirements.txt
```

---

## ▶️ **How to Run the Project**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/Big-Mart-Sales-Prediction.git
cd Big-Mart-Sales-Prediction
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Notebook or Scripts**

You can either:

* Open the Jupyter Notebook or Python File

```bash
jupyter notebook notebooks/big_mart_analysis.ipynb
```

**OR**


```bash
python big_mart_analysis.py
```

---

## 📊 **Models Used**

* Linear Regression
* Ridge/Lasso Regression
* Random Forest Regressor
* XGBoost Regressor
* LightGBM (optional)
* Stacking/Ensembling (optional)

---

## 🏆 **Final Deliverables**

* **Prediction CSV** in required format
* **Complete Code** enabling reproducibility
* **Documentation (this README)**

---

## 🤝 **Contributions**

Feel free to fork the repo & contribute via pull requests.
Suggestions and improvements are always welcome!
