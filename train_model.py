import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle
import os

DATA_PATH = "data/sms_spam_sample.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
X, y = df["message"], df["label"].map({"ham": 0, "spam": 1})

print("Training model...")
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vec, y)

print("Saving model...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved as model.pkl and vectorizer.pkl")
