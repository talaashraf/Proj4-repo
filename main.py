from preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split
from features import create_bow_features, create_tfidf_features
from models import train_naive_bayes, train_logistic_regression
from evaluation import evaluate_model
from embeddings import create_embeddings
from tuning import tune_logistic_regression
from deployment import save_model
from visualization import compare_models
import joblib

df = load_and_preprocess()
# 3. Display basic dataset information before cleaning
print("First 5 rows before cleaning:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

# --------------------------------
# Train/Test Split
# --------------------------------

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)


# --------------------------------
# Bag of Words
# --------------------------------

X_train_bow, X_test_bow, bow_vectorizer = create_bow_features(
    X_train_text,
    X_test_text
)

print("\nBag of Words Training Shape:")
print(X_train_bow.shape)

print("\nBag of Words Testing Shape:")
print(X_test_bow.shape)

print("\nFirst 10 Bag of Words Features:")
print(bow_vectorizer.get_feature_names_out()[:10])


# --------------------------------
# TF-IDF
# --------------------------------

X_train_tfidf, X_test_tfidf, tfidf_vectorizer = create_tfidf_features(
    X_train_text,
    X_test_text
)
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")

print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF Testing Shape:")
print(X_test_tfidf.shape)

print("\nFirst 10 TF-IDF Features:")
print(tfidf_vectorizer.get_feature_names_out()[:10])


# Use TF-IDF features for model training
X_train = X_train_tfidf
X_test = X_test_tfidf
# -------------------------------
# Naive Bayes
# -------------------------------

nb_model = train_naive_bayes(X_train, y_train)

nb_predictions = nb_model.predict(X_test)


# -------------------------------
# Logistic Regression
# -------------------------------

lr_model = train_logistic_regression(X_train, y_train)
joblib.dump(lr_model, "sentiment_model.pkl")

lr_predictions = lr_model.predict(X_test)

# Evaluate Naive Bayes
nb_accuracy, nb_precision, nb_recall, nb_f1 = evaluate_model(
    "Naive Bayes",
    y_test,
    nb_predictions
)


# Evaluate Logistic Regression
lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)
# --------------------------------
# Model Tuning
# --------------------------------

tuned_lr_model = tune_logistic_regression(
    X_train,
    y_train
)

tuned_lr_predictions = tuned_lr_model.predict(X_test)

tuned_accuracy, tuned_precision, tuned_recall, tuned_f1 = evaluate_model(
    "Tuned Logistic Regression",
    y_test,
    tuned_lr_predictions
)
print("\n========== Before vs After Tuning ==========")

print("\nOriginal Logistic Regression")
print("Accuracy:", lr_accuracy)
print("F1-score:", lr_f1)

print("\nTuned Logistic Regression")
print("Accuracy:", tuned_accuracy)
print("F1-score:", tuned_f1)

if tuned_f1 > lr_f1:
    print("\nTuning improved the model.")
elif tuned_f1 < lr_f1:
    print("\nOriginal model performed better.")
else:
    print("\nBoth models achieved the same F1-score.")
# --------------------------------
# Embeddings
# --------------------------------

X_train_emb, X_test_emb = create_embeddings(
    X_train_text,
    X_test_text
)

embedding_model = train_logistic_regression(
    X_train_emb,
    y_train
)

save_model(
    embedding_model,
    "embedding_logistic_model.pkl"
)

embedding_predictions = embedding_model.predict(
    X_test_emb
)

emb_accuracy, emb_precision, emb_recall, emb_f1 = evaluate_model(
    "Logistic Regression + Embeddings",
    y_test,
    embedding_predictions
)

# Feature Comparison

print("\n========== Feature Comparison ==========")

print("\nTF-IDF + Logistic Regression")
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1-score:", lr_f1)

print("\nEmbeddings + Logistic Regression")
print("Accuracy:", emb_accuracy)
print("Precision:", emb_precision)
print("Recall:", emb_recall)
print("F1-score:", emb_f1)

if emb_f1 > lr_f1:
    print("\nEmbeddings performed better based on F1-score.")

elif lr_f1 > emb_f1:
    print("\nTF-IDF performed better based on F1-score.")

else:
    print("\nBoth methods achieved the same F1-score.")

compare_models(
    nb_f1,
    lr_f1,
    emb_f1
)