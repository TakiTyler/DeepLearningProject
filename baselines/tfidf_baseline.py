"""TF-IDF retrieval baseline.

Fits a TF-IDF vectorizer on the training split's ingredient strings,
then at query time returns the top-1 most-similar training recipe as
the "generation." Mirrors the KNN retrieval approach from the RRS
related-work system and provides a non-LLM reference point.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from splits import build_splits
from inference import format_reference
from evaluate import evaluate_model


def _ingredient_text(ingredients):
    return " ".join(ingredients)


def build_retriever(train_ds):
    corpus = [_ingredient_text(ex["ingredients"]) for ex in train_ds]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)

    # pre-format each training recipe as its "generation" text
    formatted = [
        format_reference(ex["name"], ex["steps"]) for ex in train_ds
    ]

    def generate_fn(ingredients):
        query_vec = vectorizer.transform([_ingredient_text(ingredients)])
        sims = cosine_similarity(query_vec, matrix)[0]
        best = int(sims.argmax())
        return formatted[best]

    return generate_fn


def main(output_dir="."):
    train, _val, test = build_splits()
    gen_fn = build_retriever(train)
    evaluate_model(gen_fn, test, run_name="tfidf", rank="-", output_dir=output_dir)


if __name__ == "__main__":
    # This allows running just this baseline from the command line.
    main()
