import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

# Load and clean data
df = pd.read_csv("movie.csv")
df.dropna(inplace=True)

# Extract only the main genre (first in the list)
df['main_genre'] = df['genres'].apply(lambda x: x.split('|')[0])

# Inputs and target
X = df['title']
y = df['main_genre']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------- WORD CLOUD ---------
all_titles = " ".join(X)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Movie Titles")
plt.tight_layout()
plt.show()

# --------- Bag-of-Words ---------
print("\n📘 Training with Bag-of-Words")
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

model_bow = MultinomialNB()
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)

print("\n✅ BoW Results")
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))

cm_bow = confusion_matrix(y_test, y_pred_bow, labels=model_bow.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_bow, annot=True, fmt='d', cmap='Blues',
            xticklabels=model_bow.classes_, yticklabels=model_bow.classes_)
plt.title("Confusion Matrix (BoW)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# --------- TF-IDF ---------
print("\n📗 Training with TF-IDF")
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model_tfidf = MultinomialNB()
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

print("\n✅ TF-IDF Results")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))

cm_tfidf = confusion_matrix(y_test, y_pred_tfidf, labels=model_tfidf.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Greens',
            xticklabels=model_tfidf.classes_, yticklabels=model_tfidf.classes_)
plt.title("Confusion Matrix (TF-IDF)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
