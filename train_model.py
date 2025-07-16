# train_model_pipeline.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 1. Load Raw Data
df = pd.read_csv("customer_churn_data.csv")

# 2. Data Cleaning
# Drop duplicates and handle missing values
df = df.drop_duplicates()
thresh = len(df) * 0.5

df = df.dropna(thresh=thresh, axis=1)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# 3. Split Features & Target
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn', 'CustomerID'])

# 4. Identify Column Types
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include='number').columns.tolist()

# 5. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# 6. Final Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Model
pipeline.fit(X_train, y_train)

# 9. Save Entire Pipeline
joblib.dump(pipeline, 'model.pkl')
print("✅ Model pipeline saved as model.pkl")

# 10. Evaluate
preds = pipeline.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, preds))

sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
