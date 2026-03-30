import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

FEATURE_TYPE = "tfidf_uni_bi"

MODELS = {
    "1": "logreg",
    "2": "svm",
    "3": "nb",
    "4": "rf"
}

print("\n========== SENTIMENT ANALYSIS SYSTEM ==========")
print("\nChoose Model:")
print("1 - Logistic Regression")
print("2 - Support Vector Machine")
print("3 - Naive Bayes")
print("4 - Random Forest")

choice = input("\nEnter your choice (1-4): ")

if choice not in MODELS:
    print("Invalid choice.")
    exit()

model_name = MODELS[choice]

model_path = f"models/{FEATURE_TYPE}_{model_name}.pkl"
vectorizer_path = f"models/{FEATURE_TYPE}_{model_name}_vectorizer.pkl"

if not os.path.exists(model_path):
    print("Model file not found. Train models first.")
    exit()

# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

review = input("\nEnter a movie review:\n")

# Transform input
review_vec = vectorizer.transform([review])

# Predict
prediction = model.predict(review_vec)[0]
sentiment = "Positive" if prediction == 1 else "Negative"

print("\n=========== RESULT ===========")
print("Model Used:", model_name.upper())
print("Predicted Sentiment:", sentiment)

# Confidence Handling
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(review_vec)[0]
    confidence = max(prob)
    print("Confidence:", round(confidence, 4))

elif hasattr(model, "decision_function"):
    score = model.decision_function(review_vec)[0]
    confidence = 1 / (1 + np.exp(-score))
    print("Decision Score:", round(score, 4))
    print("Confidence:", round(confidence, 4))

# - Feature Importance (Only Linear Models) -
if model_name in ["logreg", "svm"]:
    print("\nTop Influential Words:\n")

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    indices = review_vec.nonzero()[1]
    contributions = []

    for idx in indices:
        word = feature_names[idx]
        contribution = coefficients[idx] * review_vec[0, idx]
        contributions.append((word, contribution))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    for word, score in contributions[:10]:
        print(f"{word} : {round(score,4)}")

    # Plot graph
    top_words = [w[0] for w in contributions[:10]]
    top_scores = [w[1] for w in contributions[:10]]

    plt.figure()
    plt.barh(top_words, top_scores)
    plt.title(f"Top Word Contributions ({model_name.upper()})")
    plt.xlabel("Contribution Score")
    plt.gca().invert_yaxis()
    plt.show()

else:
    print("\nThis model does not provide direct word coefficients.")
    print("It uses probabilistic or tree-based decision logic.")

print("\n================================")