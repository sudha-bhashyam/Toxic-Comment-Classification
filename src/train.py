import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from data_preprocessing import clean_text

# Load dataset
df = pd.read_csv('../data/train.csv')

# Keep only required columns
df = df[['comment_text', 'toxic']]

# Clean text
df['cleaned'] = df['comment_text'].apply(clean_text)

# Features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['toxic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, '../model/model.pkl')
joblib.dump(vectorizer, '../model/vectorizer.pkl')

print("\nModel and vectorizer saved successfully!")