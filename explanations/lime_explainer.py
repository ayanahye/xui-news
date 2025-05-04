import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
categories = df['category'].unique()
selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
df_sampled = df[df['category'].isin(selected_categories)].sample(n=500, random_state=42)

X = df_sampled['combined_text']
y = df_sampled['category']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

explainer = LimeTextExplainer(class_names=class_names)

_pipeline_cache = {}

def get_lime_pipeline(min_df=5):
    if min_df in _pipeline_cache:
        return _pipeline_cache[min_df]
    vectorizer = TfidfVectorizer(min_df=min_df)
    pipeline = make_pipeline(vectorizer, RandomForestClassifier())
    pipeline.fit(X, y_encoded)
    _pipeline_cache[min_df] = (pipeline, vectorizer)
    return pipeline, vectorizer

# override the default min df and num features with user provided
def explain_with_lime(text_instance, min_df=5, num_features=6):
    pipeline, vectorizer = get_lime_pipeline(min_df)
    # https://lime-ml.readthedocs.io/en/latest/lime.html
    # additional params could add: num_samples (size of neighborhood to learn the linear model)
    # potentially distance metrics
    exp = explainer.explain_instance(
        text_instance, pipeline.predict_proba, num_features=num_features
    )
    html = exp.as_html()
    prediction = pipeline.predict([text_instance])[0]
    predicted_class = le.inverse_transform([prediction])[0]
    return {"visualization": html, "predicted_class": predicted_class}
