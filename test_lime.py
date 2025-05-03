import pandas as pd
import numpy as np
import shap
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from IPython.display import display, HTML

df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')

categories = df['category'].unique()
selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
df_sampled = df[df['category'].isin(selected_categories)].sample(n=500, random_state=42)

X = df_sampled['combined_text']
y = df_sampled['category']

vectorizer = TfidfVectorizer(min_df=5)
X_vec = vectorizer.fit_transform(X).toarray()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
X_train_text, X_test_text, _, _ = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

ind = 6
text = X_test_text.iloc[ind]  
print(f"original text at index {ind}:\n{text}")

model = RandomForestClassifier()
pipeline = make_pipeline(TfidfVectorizer(min_df=5), model)
pipeline.fit(X_train_text, y_train)

class_names = le.classes_
explainer = LimeTextExplainer(class_names=class_names)

ind = 6
text_instance = X_test_text.iloc[ind] 

exp = explainer.explain_instance(text_instance, pipeline.predict_proba, num_features=6)
exp.save_to_file('lime_explanation.html')

true_label = le.inverse_transform([y_test[ind]])[0]
predicted_label = le.inverse_transform([pipeline.predict([text_instance])[0]])[0]
is_correct = predicted_label == true_label

print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")
print(f"Prediction {'correct' if is_correct else 'incorrect'}")

