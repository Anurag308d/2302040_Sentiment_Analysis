import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# Set file paths for the dataset
data_path = "../data"
positive_data_path = os.path.join(data_path, "rt-polarity.pos")
negative_data_path = os.path.join(data_path, "rt-polarity.neg")

# Load positive and negative reviews
with open(positive_data_path, "r", encoding="ISO-8859-1") as pos_file:
    pos_reviews = pos_file.readlines()
with open(negative_data_path, "r", encoding="ISO-8859-1") as neg_file:
    neg_reviews = neg_file.readlines()

# Create labels for positive and negative reviews: positive = 1, negative = 0
pos_labels = [1] * len(pos_reviews)
neg_labels = [0] * len(neg_reviews)

# Combine reviews and labels
all_reviews = pos_reviews + neg_reviews
all_labels = pos_labels + neg_labels

# Split dataset into train, validation, and test sets
train_reviews = all_reviews[:4000] + all_reviews[5331:9331]
train_labels = all_labels[:4000] + all_labels[5331:9331]

val_reviews = all_reviews[4000:4500] + all_reviews[9331:9831]
val_labels = all_labels[4000:4500] + all_labels[9331:9831]

test_reviews = all_reviews[4500:5331] + all_reviews[9831:]
test_labels = all_labels[4500:5331] + all_labels[9831:]

# Convert reviews to TF-IDF feature vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(train_reviews)
X_val_tfidf = tfidf_vectorizer.transform(val_reviews)
X_test_tfidf = tfidf_vectorizer.transform(test_reviews)

# Define the model and parameter grid for hyperparameter tuning
param_options = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
naive_bayes_classifier = MultinomialNB()
grid_search = GridSearchCV(naive_bayes_classifier, param_options, cv=5, scoring='accuracy')

# Train the model using GridSearchCV for the best hyperparameters
grid_search.fit(X_train_tfidf, train_labels)

# Get the best model after tuning
best_model = grid_search.best_estimator_
print(f"Best Parameters Found: {grid_search.best_params_}")

# Validation accuracy using the best model
val_accuracy = best_model.score(X_val_tfidf, val_labels)
print(f"Validation Accuracy with Best Model: {val_accuracy:.2f}")

# Evaluate the model on the test set
test_predictions = best_model.predict(X_test_tfidf)

# Calculate confusion matrix and other metrics
tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()

# Calculate Precision, Recall, and F1 Score
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Display metrics
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
