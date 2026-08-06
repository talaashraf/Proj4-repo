import joblib
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text
from database_operations import create_prediction, get_all_predictions

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

classifier = joblib.load(
    "embedding_logistic_model.pkl"
)


def predict_sentiment(text):
    cleaned_text = clean_text(text)

    text_embedding = embedding_model.encode(
        [cleaned_text]
    )

    prediction = classifier.predict(
        text_embedding
    )
    create_prediction(
    input_text=text,
    predicted_sentiment=prediction[0]
           )
    return prediction[0]


user_text = input("Enter a text: ")

result = predict_sentiment(user_text)

print("Predicted Sentiment:", result)
print("\nPrediction History:")

for row in get_all_predictions():
    print(row)