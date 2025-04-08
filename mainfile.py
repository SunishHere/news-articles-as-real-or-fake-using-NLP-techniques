import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset (tiny version for demo)
data = {
    "text": [
        "The government has approved new policies to boost the economy.",
        "Breaking: Actor caught in scandal that never happened!",
        "Scientists confirm water on Mars.",
        "You won't believe what this minister did! Click to know more.",
        "New study reveals the benefits of meditation.",
        "Shocking news! Vaccines are harmful, says anonymous source.",
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
}
df = pd.DataFrame(data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.33, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {accuracy*100:.2f}%)")
conf_matrix_path = "/mnt/data/Fake_News_Confusion_Matrix.png"
plt.tight_layout()
plt.savefig(conf_matrix_path)
plt.close()

conf_matrix_path
