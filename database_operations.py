from database import engine, sentiment_prediction


def create_prediction(
    input_text,
    predicted_sentiment,
    prediction_score=None
):
    query = sentiment_prediction.insert().values(
        input_text=input_text,
        predicted_sentiment=predicted_sentiment,
        prediction_score=prediction_score
    )

    with engine.begin() as connection:
        connection.execute(query)


def get_all_predictions():
    query = sentiment_prediction.select().order_by(
        sentiment_prediction.c.id.desc()
    )

    with engine.connect() as connection:
        result = connection.execute(query)
        return result.fetchall()