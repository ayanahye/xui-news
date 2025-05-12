import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from tqdm import tqdm

class LOOValuation:
    def __init__(self, data_path="News_Category_Dataset_v3.json"):
        self.data_path = data_path

    def compute_loo_values(self):
        df = pd.read_json(self.data_path, lines=True)
        df['combined_text'] = (
            df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
        )
        categories = df['category'].unique()
        selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
        df_sampled = df[df['category'].isin(selected_categories)].sample(n=500, random_state=42)

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

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_vec, y_train)
        baseline_val_proba = model.predict_proba(X_val_vec)
        baseline_val_logloss = log_loss(y_val, baseline_val_proba)

        print(f"baseline val log-loss: {baseline_val_logloss:.4f}")

        loo_values = []
        for i in tqdm(range(len(X_train_vec)), desc="Computing LOO Values", ncols=100):
            X_train_loo = np.delete(X_train_vec, i, axis=0)
            y_train_loo = np.delete(y_train, i)
            
            model_loo = RandomForestClassifier(random_state=42)
            model_loo.fit(X_train_loo, y_train_loo)
            
            val_proba_loo = model_loo.predict_proba(X_val_vec)
            loo_val_logloss = log_loss(y_val, val_proba_loo)

            loo_value = baseline_val_logloss - loo_val_logloss
            loo_values.append(loo_value)

            print(f"index: {i}, LOO value: {loo_value:.6f}")

        return loo_values

if __name__ == "__main__":
    loo_valuation = LOOValuation(data_path="News_Category_Dataset_v3.json")
    loo_values = loo_valuation.compute_loo_values()
    # print(loo_values)

# let them choose utility function
# using log loss now which indicates how close the prediction probability is the corresponding actual true value
# high loss is bad
# so positive values means left out point was helpful 
# negative values means bad 