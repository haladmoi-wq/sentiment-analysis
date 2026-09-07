Sentiment Analysis

Overview

This project uses Machine Learning and Natural Language Processing (NLP) to classify movie reviews as positive or negative.

The model is trained on the IMDb Dataset of 50,000 movie reviews.

Technologies

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression

Dataset

The project uses the IMDb Movie Reviews dataset, containing 50,000 labeled movie reviews.

The dataset is not included in this repository because of its size. The .gitignore file prevents the dataset from being uploaded to GitHub.

Method

The project follows these steps:

1. Load the movie review dataset.
2. Convert text into numerical features using TF-IDF.
3. Split the data into training and testing sets.
4. Train a Logistic Regression model.
5. Predict the sentiment of new movie reviews.
6. Evaluate the model using accuracy.

Results

The model achieved approximately 90% accuracy on the test set.

Example predictions:

* “I absolutely loved this movie.” → Positive
* “The movie was boring and I did not enjoy it.” → Negative

How to Run

Install the required libraries:

pip install pandas scikit-learn

Place the IMDb dataset file named IMDB Dataset.csv in the project folder, then run:

python sentiment_analysis.py

Enter a movie review when prompted to receive a sentiment prediction.

Purpose

This project was created to practice Natural Language Processing, text classification, and Machine Learning.

It is part of my Artificial Intelligence portfolio.