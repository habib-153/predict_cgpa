import pandas as pd
import numpy as np

# sklearn preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Regression model
from sklearn.ensemble import RandomForestRegressor

import pickle

# metrices

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./bangladesh_student_performance.csv')
if 'date' in df.columns:
  df.drop(columns=['date'], inplace=True)
X = df.drop('hsc_result', axis=1)
y = df['hsc_result']

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Numerical Feature : {numerical_features}")
print(f"Categorical Feature : {categorical_features}")

num_tranformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)
# Categorical
cat_tranformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)
# combine them
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_tranformer, numerical_features),
        ('cat', cat_tranformer, categorical_features)
    ]
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf_pipeline = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, min_samples_split=2, max_depth=10, random_state=42, n_jobs=-1))
    ]
)

rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Random Forest Regressor Performance:")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
print(f"MAE: {mae}")

# Save the model
with open('student_performance_model.pkl', 'wb') as f:
  pickle.dump(rf_pipeline, f)

print("Model saved as student_performance_model.pkl")