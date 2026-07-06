# ML-Weather-Forecast
This repository contains an end‑to‑end machine learning workflow for forecasting weather conditions using a Kaggle dataset focused on Indian meteorological data. This project includes data preprocessing, exploratory analysis, feature engineering, model training, evaluation, and forecasting using classical ML techniques. 

Dataset: Kaggle India Weather (Jan–Oct 2022, 7,056 hourly observations). Cleaned, unused columns removed, timestamp → DatetimeIndex.

EDA: Weekly rolling mean, ACF/PACF lag analysis, seasonal pattern inspection.

Feature Engineering: Rolling window (14 lag hours as inputs, next-hour temperature as target) converting time series to supervised learning.

Model Pipeline: RobustScaler + XGBRegressor. Tuned via GridSearchCV & RandomizedSearchCV. Best params: lr=0.01, max_depth=4, n_estimators=400, subsample=0.7.

Results: MAE ~1.54, MSE ~6.78, MAPE ~23.6%. Stable forecasts with no extreme outliers.
