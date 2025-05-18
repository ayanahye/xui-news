import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from beta_shapley.betashap.ShapEngine import ShapEngine

class BetaShapExplainer:
    def __init__(self, data_path="News_Category_Dataset_v3.json"):
        self.data_path = data_path

    def compute_leaderboard(self, alpha=16, beta=1, permutations=2, utility='likelihood',
                            min_df=5, ngram_range=(1,1), train_data_size=1000, features_to_remove=None, remove_stopwords=True):
        df = pd.read_json(self.data_path, lines=True)
        df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
        categories = df['category'].unique()
        selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
        num_cats = len(selected_categories)
        per_cat = train_data_size // num_cats
        dfs = []
        for cat in selected_categories:
            df_cat = df[df['category'] == cat].sample(n=per_cat, random_state=42)
            dfs.append(df_cat)
        df_sampled = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

        all_indices = np.arange(len(df_sampled))
        y = df_sampled['category']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        idx_temp, idx_test, y_temp, y_test = train_test_split(
            all_indices, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )

        df_sampled.to_csv("data_used/sampled_data_used.csv", index=False)
        np.savez("data_used/split_indices.npz", idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

        X = df_sampled['combined_text'].reset_index(drop=True)
        y = df_sampled['category'].reset_index(drop=True)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train_text = X.iloc[idx_train].reset_index(drop=True)
        y_train = y_encoded[idx_train]
        X_val_text = X.iloc[idx_val].reset_index(drop=True)
        y_val = y_encoded[idx_val]
        X_test_text = X.iloc[idx_test].reset_index(drop=True)
        y_test = y_encoded[idx_test]

        pd.DataFrame({'text': X_train_text, 'label': y_train}).to_csv("data_used/train_split.csv", index=False)
        pd.DataFrame({'text': X_val_text, 'label': y_val}).to_csv("data_used/val_split.csv", index=False)
        pd.DataFrame({'text': X_test_text, 'label': y_test}).to_csv("data_used/test_split.csv", index=False)

        vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, stop_words='english' if remove_stopwords else None)
        X_train_vec = vectorizer.fit_transform(X_train_text).toarray()
        X_val_vec = vectorizer.transform(X_val_text).toarray()
        if features_to_remove:
            feature_names = vectorizer.get_feature_names_out()
            keep_indices = [i for i, f in enumerate(feature_names) if f not in features_to_remove]
            X_train_vec = X_train_vec[:, keep_indices]
            X_val_vec = X_val_vec[:, keep_indices]

        engine = ShapEngine(
            X_train_vec, y_train, X_val_vec, y_val,
            problem='classification',
            model_family='logistic',
            metric=utility, 
            max_iters=permutations,
            seed=42
        )

        engine.run(knn_run=False, loo_run=True, weights_list=[(alpha, beta)])

        beta_key = f'Beta({beta},{alpha})'
        beta_shapley = engine.results[beta_key]
        loo_shapley = engine.results['LOO-Last']

        leaderboard = []
        for i in range(len(X_train_text)):
            leaderboard.append({
                "index": int(idx_train[i]),
                "text": str(X_train_text.iloc[i]),
                "label": str(le.inverse_transform([y_train[i]])[0]),
                "beta_shapley": float(beta_shapley[i]),
                "loo": float(loo_shapley[i])
            })

        leaderboard = sorted(leaderboard, key=lambda x: x["beta_shapley"], reverse=True)
        curr_vals = pd.DataFrame(leaderboard)
        curr_vals.to_csv("leaderboard.csv", index=False)
        
        return leaderboard
