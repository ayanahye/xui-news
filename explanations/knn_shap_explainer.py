# p much the same code from test_knn_shapley.py
# ignore this script for now and the module 3 overall
# seems like values change on every run and no option for random state passing..
# have to look more into the algorithm of knnshapleyvaluation 
# otherwise will need another technique for better approximation maybe..

# for now can just be a concept

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from pydvl.valuation import Dataset
from pydvl.valuation.methods import KNNShapleyValuation
import random

class KNNShapExplainer:
    def __init__(self, data_path="News_Category_Dataset_v3.json"):
        self.data_path = data_path

    def compute_leaderboard(self):
        np.random.seed(42)
        random.seed(42)

        df = pd.read_json(self.data_path, lines=True)
        df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')

        categories = df['category'].unique()
        selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
        df_sampled = df[df['category'].isin(selected_categories)].sample(n=500, random_state=42)

        X = df_sampled['combined_text'].reset_index(drop=True)
        y = df_sampled['category'].reset_index(drop=True)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        vectorizer = TfidfVectorizer(min_df=5)
        X_train_vec = vectorizer.fit_transform(X_train_text).toarray()
        X_test_vec = vectorizer.transform(X_test_text).toarray()

        feature_names = [f"f{i}" for i in range(X_train_vec.shape[1])]
        train_ds, _ = Dataset.from_arrays(X_train_vec, np.array(y_train), feature_names=feature_names)
        test_ds, _ = Dataset.from_arrays(X_test_vec, np.array(y_test), feature_names=feature_names)

        knn = KNeighborsClassifier(n_neighbors=5)
        valuation = KNNShapleyValuation(model=knn, test_data=test_ds, progress=False)
        valuation.fit(train_ds)
        # just need to clarify, because this looks at absolute value, but im assuming higher shapley value = more value so negative would be bad.
        result = valuation.result.sort()  

        indices = result.indices[::-1]
        values = result.values[::-1]

        leaderboard = []
        for i in range(len(values)):
            idx = indices[i]
            leaderboard.append({
                "index": int(idx),
                "value": float(values[i]),
                "text": str(X_train_text.iloc[idx]),
                "label": str(le.inverse_transform([y_train[idx]])[0])
            })
        return leaderboard


