import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GenresBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.unique_genres = sorted(set(g for sublist in X for g in sublist))
        return self

    def transform(self, X):
        df = pd.DataFrame(0, index=range(len(X)), columns=self.unique_genres)
        for i, genres in enumerate(X):
            for g in genres:
                if g in df.columns:
                    df.at[i, g] = 1
        return df

class SynopsisVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)
