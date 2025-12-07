# ===========================
# BigMart Sales Prediction
# High Performance
# ===========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Load Data
# ---------------------------
train = pd.read_csv("./ABB/train.csv")
test = pd.read_csv("./ABB/test.csv")

# ---------------------------
# Basic Cleaning / FE
# ---------------------------

# Standardize Item_Fat_Content
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(
    {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
)
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(
    {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
)

# Create outlet age
train['Outlet_Age'] = 2013 - train['Outlet_Establishment_Year']
test['Outlet_Age'] = 2013 - test['Outlet_Establishment_Year']

# Replace zero visibility with mean
mean_visibility = train[train['Item_Visibility'] > 0]['Item_Visibility'].mean()
train.loc[train['Item_Visibility'] == 0, 'Item_Visibility'] = mean_visibility
test.loc[test['Item_Visibility'] == 0, 'Item_Visibility'] = mean_visibility

# Extract item type from Item_Identifier (FD, DR, NC)
train['Item_Category'] = train['Item_Identifier'].str[:2]
test['Item_Category'] = test['Item_Identifier'].str[:2]

# ---------------------------
# Features
# ---------------------------
target = "Item_Outlet_Sales"

num_features = [
    "Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age"
]

cat_features = [
    "Item_Fat_Content", "Item_Type", "Outlet_Identifier",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Item_Category"
]

X = train[num_features + cat_features]
y = train[target]
X_test_final = test[num_features + cat_features]

# ---------------------------
# Preprocessing
# ---------------------------
numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# ---------------------------
# Model (Simple + Strong)
# ---------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', model)
])

# ---------------------------
# CV Check
# ---------------------------
scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE:", -scores.mean())

# ---------------------------
# Train Final Model
# ---------------------------
pipe.fit(X, y)

# ---------------------------
# Predict on Test
# ---------------------------
preds = pipe.predict(X_test_final)

# ---------------------------
# Save Submission
# ---------------------------
submission = pd.DataFrame({
    "Item_Identifier": test["Item_Identifier"],
    "Outlet_Identifier": test["Outlet_Identifier"],
    "Item_Outlet_Sales": preds
})

submission.to_csv("submission_bigmart.csv", index=False)
print("Saved submission.csv")
