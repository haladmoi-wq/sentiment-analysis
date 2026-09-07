import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("IMDB Dataset.csv")

# Convert reviews into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["review"])
y = data["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Test the model with new text
user_text = input("Enter a movie review: ")

new_text = vectorizer.transform([user_text])
prediction = model.predict(new_text)

print("Predicted sentiment:", prediction[0])

# Calculate model accuracy
accuracy = model.score(X_test, y_test)

print("Model accuracy:", accuracy)