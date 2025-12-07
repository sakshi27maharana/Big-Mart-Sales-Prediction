# BigMart — 1-Page Approach & Experimentation Note

**Project goal.** Predict `Item_Outlet_Sales` for unseen item–outlet pairs (train: 8,523 rows; test: 5,681 rows) and produce an RMSE-optimized submission. The delivered script is a compact, reproducible pipeline that moves from raw CSVs → cleaned features → model → submission.

---

## Summary of steps taken

1. **Data load**

   * Read `train.csv` and `test.csv`. Kept column names and original indices for later submission mapping.

2. **Quick, targeted cleaning & feature engineering**

   * **Standardized `Item_Fat_Content`** labels (`LF`, `low fat` → `Low Fat`; `reg` → `Regular`) to avoid fragmented categories.
   * **Outlet age**: created `Outlet_Age = 2013 - Outlet_Establishment_Year` to capture store maturity effects.
   * **Zero visibility fix**: replaced `Item_Visibility == 0` using the mean visibility from non-zero training rows — zeros are likely reporting glitches.
   * **Item category**: extracted first two characters of `Item_Identifier` (e.g., `FD`, `DR`, `NC`) as `Item_Category` to capture coarse product groups.
   * **Kept numerical features**: `Item_Weight`, `Item_Visibility`, `Item_MRP`, `Outlet_Age`.
   * **Categorical set**: `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`, `Item_Category`.

3. **Preprocessing pipeline**

   * **Numerical**: median imputation (`SimpleImputer(strategy='median')`) — robust to skew and outliers.
   * **Categorical**: most frequent imputation, then `OneHotEncoder(handle_unknown='ignore')`. Using `sparse_output=False` yields a dense array ready for scikit pipelines.
   * **ColumnTransformer** used to keep transformations modular and reproducible.

4. **Model choice**

   * Final model: `RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)`.
   * Chosen for a balance of strong baseline performance, minimal preprocessing required, natural handling of mixed feature types after encoding, and interpretability via feature importance.

5. **Validation & training**

   * 5-fold cross-validation using `cross_val_score` with `scoring='neg_root_mean_squared_error'`. Report CV RMSE as `-scores.mean()` to align with RMSE (lower is better).
   * After CV check, the pipeline is fit on the full training set and used to produce predictions on `test.csv`.

6. **Submission**

   * Built `final_submission_bigmart.csv` with required columns: `Item_Identifier`, `Outlet_Identifier`, `Item_Outlet_Sales`.

---

## Rationale for key choices

* **Mean replace for zero visibility**: zeros likely mean missing reporting rather than true zero exposure; mean of non-zeros is a pragmatic fix to preserve distribution.
* **Median numeric imputation**: protects against skew/outliers in weight/MRP.
* **One-hot encoding**: keeps model expressive for categorical interactions; `handle_unknown='ignore'` prevents failures for categories present only in test.
* **Random forest**: robust baseline that tends to perform well out-of-the-box and provides a dependable CV metric before moving to heavier tuning.

---

## Short experimentation log & next steps

* Current script produces a solid reproducible baseline. To push performance further, prioritise:

  1. **Target transform**: try `log1p` on `Item_Outlet_Sales` for modeling then `expm1` back to scale — often reduces RMSE for skewed targets.
  2. **Advanced FE**: MSRP bucketing, price per weight, category×outlet interaction counts, outlet sales aggregates (groupby features).
  3. **Modeling**: run LightGBM / XGBoost with hyperparameter search (Bayesian/RandomizedSearchCV) and stacking ensembles.
  4. **Imputation refinement**: model-based imputation for `Item_Weight`, and treat `Item_Visibility` zeros via predictive imputation.
  5. **Calibration & ensembling**: blend RF + GBDT and calibrate using CV out-of-fold predictions to reduce variance.
  6. **Feature selection / regularization**: reduce dimensionality after one-hot to avoid overfitting (targeted PCA or feature hashing for high-cardinality outlets/items).

---

## Reproducibility & notes

* Random seed fixed (`random_state=42`).
* Paths are relative (`./train.csv` & `./test.csv`) — adjust as needed.
* Output: `final_submission.csv`.
