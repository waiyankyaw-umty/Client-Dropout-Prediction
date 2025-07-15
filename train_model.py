# churn_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load Data
df = pd.read_csv("customer_churn_data.csv")

# 2. Preview
print("📄 First 5 rows:")
print(df.head())
print("\n📊 Column Info:")
print(df.info())
print("\n🔍 Null Values:")
print(df.isnull().sum())

# 3. Basic Statistics
print("\n📈 Summary Stats:")
print(df.describe())

# 4. Churn Rate Overview
if 'Churn' in df.columns:
    print("\n✅ Churn Value Counts:")
    print(df['Churn'].value_counts())
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.show()
else:
    print("❗️ No 'Churn' column found.")

# 5. Visualize Numerical Feature Distributions
num_cols = df.select_dtypes(include='number').columns

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

# 6. Correlation Matrix (Optional if many numeric features)
if len(num_cols) >= 2:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


# Check for duplicate rows
print("\n🧹 Duplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Drop columns with too many nulls (threshold 50%)
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

# Fill remaining missing values (categorical: mode, numeric: median)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Churn vs Categorical Features
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if col != 'Churn':
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x=col, hue='Churn')
        plt.title(f'Churn vs {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Churn vs Numeric Features
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y=col)
    plt.title(f'{col} by Churn')
    plt.tight_layout()
    plt.show()

# Encode categorical values
df_encoded = pd.get_dummies(df, drop_first=True)

# Save cleaned data
df_encoded.to_csv("cleaned_customer_churn.csv", index=False)
print("✅ Cleaned data saved to cleaned_customer_churn.csv")


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Split Data
X = df_encoded.drop(columns=['Churn_Yes'])  # or df_encoded['Churn'] if binary
y = df_encoded['Churn_Yes']  # adjust based on your dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import numpy as np

# Show important features
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features for Churn")
plt.show()

