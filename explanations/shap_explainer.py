import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
categories = df['category'].unique()
selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
df_sampled = df[df['category'].isin(selected_categories)].sample(n=500, random_state=42)

X = df_sampled['combined_text']
y = df_sampled['category']

vectorizer = TfidfVectorizer(min_df=5)
X_vec = vectorizer.fit_transform(X).toarray()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
X_train_text, X_test_text = train_test_split(X, test_size=0.2, stratify=y_encoded, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

feature_names = vectorizer.get_feature_names_out()
explainer = shap.Explainer(model, X_train, feature_names=feature_names)

class ShapExplainer:
    def explain(self, text):
        X_instance = vectorizer.transform([text]).toarray()
        prediction = model.predict(X_instance)[0]
        predicted_class = le.inverse_transform([prediction])[0]
        shap_values = explainer(X_instance)
        html = shap.plots.force(shap_values[0, :, prediction], matplotlib=False)
        return {"visualization": html, "predicted_class": predicted_class}
