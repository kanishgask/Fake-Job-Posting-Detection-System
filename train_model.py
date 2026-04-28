import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from model import train_model
from utils import preprocess_text

# Load dataset (you need to download fake job dataset from Kaggle)
df = pd.read_csv("fake_job_postings.csv")

# Preprocess text
df["text"] = df["description"].fillna("").apply(preprocess_text)

# Labels (0 = real, 1 = fake)
X = df["text"]
y = df["fraudulent"]

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = train_model(X_vectorized, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")
