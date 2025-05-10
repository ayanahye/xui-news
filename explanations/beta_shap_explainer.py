import numpy as np
from beta_shapley.betashap.ShapEngine import ShapEngine

class BetaShapExplainer:
    def __init__(self, data_path="News_Category_Dataset_v3.json"):
        self.data_path = data_path

    def compute_leaderboard(self):
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_json(self.data_path, lines=True)
        df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
        categories = df['category'].unique()
        selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
        df_sampled = df[df['category'].isin(selected_categories)].sample(n=100, random_state=42)

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

        engine = ShapEngine(
            X_train_vec, y_train, X_test_vec, y_test,
            problem='classification',
            model_family='RandomForest',
            metric='accuracy',
            max_iters=1,
            seed=42
        )
        engine.run(loo_run=True, weights_list=[(1, 1)])  # Beta(1,1) is standard Shapley

        beta_shapley = engine.results['Beta(1,1)']
        print(beta_shapley)
        loo = engine.results['LOO-Last']
        print(loo)

        leaderboard = []
        for i in range(len(X_train_text)):
            leaderboard.append({
                "index": int(i),
                "text": str(X_train_text.iloc[i]),
                "label": str(le.inverse_transform([y_train[i]])[0]),
                "beta_shapley": float(beta_shapley[i]),
                "loo": float(loo[i])
            })
        leaderboard = sorted(leaderboard, key=lambda x: x["beta_shapley"], reverse=True)
        return leaderboard
