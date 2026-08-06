from sentence_transformers import SentenceTransformer


def create_embeddings(X_train_text, X_test_text):
    """
    Convert training and testing sentences into embeddings.
    """

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("\nCreating training embeddings...")

    X_train_emb = embedding_model.encode(
        X_train_text.tolist(),
        show_progress_bar=True
    )

    print("\nCreating testing embeddings...")

    X_test_emb = embedding_model.encode(
        X_test_text.tolist(),
        show_progress_bar=True
    )

    return X_train_emb, X_test_emb