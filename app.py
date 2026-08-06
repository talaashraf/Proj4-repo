import joblib
import streamlit as st
from preprocessing import clean_text
from database_operations import create_prediction


# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


st.title("Sentiment Analysis")

st.write(
    "Enter a sentence and the model will predict "
    "whether it is positive, negative, or neutral."
)

user_text = st.text_area(
    "Enter your text:"
)

if st.button("Predict"):

    if user_text.strip() == "":
        st.warning("Please enter a sentence.")

    else:
        cleaned_text = clean_text(user_text)

        text_vector = vectorizer.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            text_vector
        )[0]
        create_prediction(
        input_text=user_text,
        predicted_sentiment=prediction
         )

        st.subheader("Predicted Sentiment")

        if prediction.lower() == "positive":
            st.success("Positive ")

        elif prediction.lower() == "negative":
            st.error("Negative ")

        else:
            st.info("Neutral ")