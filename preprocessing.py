import pandas as pd
import re

# imports تبعك
from wordfreq import zipf_frequency
from spellchecker import SpellChecker

spell = SpellChecker()


def clean_text(text):
    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    cleaned_words = []

    for word in text.split():
        if zipf_frequency(word, "en") > 0:
            cleaned_words.append(word)
        else:
            corrected = spell.correction(word)
            cleaned_words.append(corrected if corrected else word)

    return " ".join(cleaned_words)


def load_and_preprocess():

    df = pd.read_csv("data/train.csv", encoding="latin1")

    df = df[["text", "sentiment"]]

    df = df.dropna()

    df["clean_text"] = df["text"].apply(clean_text)

    print("\nBefore Cleaning:")
    print(df["text"].head(10))

    print("\nAfter Cleaning:")
    print(df["clean_text"].head(10))

    return df