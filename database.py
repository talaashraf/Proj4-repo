from sqlalchemy import ( create_engine,MetaData,Table,Column,Integer,Text,String,Float,TIMESTAMP,func)
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

metadata = MetaData()

sentiment_prediction = Table(
    "sentiment_prediction",
    metadata,

    Column("id", Integer, primary_key=True),
    Column("input_text", Text, nullable=False),
    Column("predicted_sentiment", String(20), nullable=False),
    Column("prediction_score", Float),
    Column("created_at", TIMESTAMP, server_default=func.now())
)

metadata.create_all(engine)