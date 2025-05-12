import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from beta_shapley.betashap.ShapEngine import ShapEngine

class BetaShapExplainer:
    def __init__(self, data_path="News_Category_Dataset_v3.json"):
        self.data_path = data_path

    def compute_leaderboard(self):
        df = pd.read_json(self.data_path, lines=True)
        df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
        categories = df['category'].unique()
        selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
        df_sampled = df[df['category'].isin(selected_categories)].sample(n=100, random_state=42)

        X = df_sampled['combined_text'].reset_index(drop=True)
        y = df_sampled['category'].reset_index(drop=True)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train_text, X_val_text, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        vectorizer = TfidfVectorizer(min_df=5)
        X_train_vec = vectorizer.fit_transform(X_train_text).toarray()
        X_val_vec = vectorizer.transform(X_val_text).toarray()

        engine = ShapEngine(
            X_train_vec, y_train, X_val_vec, y_val,
            problem='classification',
            model_family='RandomForest',
            metric='likelihood', 
            max_iters=2,
            seed=42
        )

        engine.run(knn_run=False, loo_run=True, weights_list=[(16, 1)])

        beta_shapley = engine.results['Beta(1,16)']
        loo_shapley = engine.results['LOO-Last']

        leaderboard = []
        for i in range(len(X_train_text)):
            leaderboard.append({
                "index": int(i),
                "text": str(X_train_text.iloc[i]),
                "label": str(le.inverse_transform([y_train[i]])[0]),
                "beta_shapley": float(beta_shapley[i]),
                "loo": float(loo_shapley[i])
            })

        leaderboard = sorted(leaderboard, key=lambda x: x["beta_shapley"], reverse=True)
        curr_vals = pd.DataFrame(leaderboard)
        curr_vals.to_csv("leaderboard.csv", index=False)
        
        return leaderboard
