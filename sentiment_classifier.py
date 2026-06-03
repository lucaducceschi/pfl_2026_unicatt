import os
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class SentimentClassifier:
    """
    A unified Sentiment Classifier pipeline for Twitter Entity Sentiment Analysis.
    Supports Count/TF-IDF vectorization and Naive Bayes / Random Forest models.
    """
    def __init__(self, vectorizer_type="count", classifier_type="naive_bayes", **kwargs):
        """
        Initialize the sentiment classifier pipeline.

        Parameters
        ----------
        vectorizer_type : str
            Type of text vectorization: "count" (CountVectorizer) or "tfidf" (TfidfVectorizer).
        classifier_type : str
            Type of classification model: "naive_bayes" (MultinomialNB) or "random_forest" (RandomForestClassifier).
        **kwargs : dict
            Additional arguments to pass to the vectorizer or classifier constructors.
            Use `vectorizer_params` (dict) or `classifier_params` (dict) to customize.
        """
        self.vectorizer_type = vectorizer_type.lower()
        self.classifier_type = classifier_type.lower()
        
        vectorizer_params = kwargs.get("vectorizer_params", {})
        classifier_params = kwargs.get("classifier_params", {})
        
        # Initialize the vectorizer
        if self.vectorizer_type == "count":
            self.vectorizer = CountVectorizer(**vectorizer_params)
        elif self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(**vectorizer_params)
        else:
            raise ValueError(f"Unsupported vectorizer type: {self.vectorizer_type}")
            
        # Initialize the classifier
        if self.classifier_type == "naive_bayes":
            self.classifier = MultinomialNB(**classifier_params)
        elif self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(**classifier_params)
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")
            
        self.is_fitted = False

    @staticmethod
    def load_raw_csv_from_zip(zip_path, file_name):
        """
        Helper method to open the Kaggle zip archive and read a CSV directly.
        Sets column names and configures quotechar to handle commas in tweets.
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Could not find ZIP archive at: {zip_path}")
            
        with ZipFile(zip_path) as arch:
            if file_name not in arch.namelist():
                raise FileNotFoundError(f"File {file_name} not found in archive {zip_path}")
            df = pd.read_csv(arch.open(file_name), header=None, quotechar='"')
            df.columns = ["ID", "ENTITY", "SENTIMENT", "TEXT"]
        return df

    def preprocess_df(self, df):
        """
        Cleans the dataframe:
        - Drops NaN values in the TEXT column
        - Filters out 'Irrelevant' sentiment labels
        Returns clean parallel lists of (texts, labels).
        """
        # Drop rows with NaN text
        cleaned_df = df.dropna(subset=["TEXT"])
        
        # Filter out 'Irrelevant' labels
        cleaned_df = cleaned_df[cleaned_df["SENTIMENT"] != "Irrelevant"]
        
        # Ensure all texts are strings (precaution against typing issues)
        texts = [str(t) for t in cleaned_df["TEXT"]]
        labels = cleaned_df["SENTIMENT"].tolist()
        
        return texts, labels

    def fit(self, train_texts, train_labels):
        """
        Fit the vectorizer and classifier on training data.
        """
        X_train = self.vectorizer.fit_transform(train_texts)
        self.classifier.fit(X_train, train_labels)
        self.is_fitted = True
        return self

    def predict(self, texts):
        """
        Predict sentiment labels for a list of text strings.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been trained yet. Call fit() first.")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def evaluate(self, val_texts, val_labels):
        """
        Predict on validation text data and calculate accuracy score.
        """
        preds = self.predict(val_texts)
        return accuracy_score(val_labels, preds)
