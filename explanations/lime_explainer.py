import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# for lime n gram doesnt actually work i think
# At the moment, we restrict the explanation terms to unigrams, even if the model is using bigrams or ngrams. However, if a particular bigram is important, LIME should highlight both words separately. I've seen this effect using bigrams, and also using LIME with models like LSTMs.
# from: https://github.com/marcotcr/lime/issues/7

_pipeline_cache = {}

def get_lime_pipeline(
    min_df=5,
    remove_stopwords=True,
    ngram_range=(1,1),
    train_data_size=300,
    features_to_remove=None
):
    features_to_remove = features_to_remove or []
    cache_key = (min_df, remove_stopwords, ngram_range, train_data_size, tuple(sorted(features_to_remove)))
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]
    df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
    df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
    categories = df['category'].unique()
    selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
    df_sampled = df[df['category'].isin(selected_categories)]
    if train_data_size is not None:
        df_sampled = df_sampled.sample(n=train_data_size, random_state=42)

    X = df_sampled['combined_text']
    y = df_sampled['category']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    stop_words = 'english' if remove_stopwords else None
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        stop_words=stop_words,
        ngram_range=ngram_range
    )
    X_vec = vectorizer.fit_transform(X).toarray()
    feature_names = vectorizer.get_feature_names_out()

    if features_to_remove:
        keep_indices = [i for i, f in enumerate(feature_names) if f not in features_to_remove]
        X_vec = X_vec[:, keep_indices]
        feature_names = feature_names[keep_indices]

    class CustomVectorizer:
        def __init__(self, vectorizer, keep_indices):
            self.vectorizer = vectorizer
            self.keep_indices = keep_indices
        def transform(self, X):
            X_vec = self.vectorizer.transform(X).toarray()
            if self.keep_indices is not None:
                X_vec = X_vec[:, self.keep_indices]
            return X_vec
        def fit(self, X, y=None):
            return self

    keep_indices = None
    if features_to_remove:
        keep_indices = [i for i, f in enumerate(vectorizer.get_feature_names_out()) if f not in features_to_remove]

    custom_vectorizer = CustomVectorizer(vectorizer, keep_indices)
    pipeline = make_pipeline(custom_vectorizer, RandomForestClassifier())
    pipeline.fit(X, y_encoded)

    _pipeline_cache[cache_key] = (pipeline, vectorizer, le, class_names, keep_indices)
    return pipeline, vectorizer, le, class_names, keep_indices

def explain_with_lime(
    text_instance,
    min_df=5,
    num_features=6,
    remove_stopwords=True,
    ngram_range=(1,1),
    train_data_size=300,
    features_to_remove=None
):
    pipeline, vectorizer, le, class_names, keep_indices = get_lime_pipeline(
        min_df, remove_stopwords, ngram_range, train_data_size, features_to_remove
    )
    explainer = LimeTextExplainer(class_names=class_names)
    # https://lime-ml.readthedocs.io/en/latest/lime.html
    # additional params could add: num_samples (size of neighborhood to learn the linear model)
    # potentially distance metrics
    exp = explainer.explain_instance(
        text_instance, pipeline.predict_proba, num_features=num_features
    )
    html = exp.as_html()
    X_instance = vectorizer.transform([text_instance]).toarray()
    if keep_indices is not None:
        X_instance = X_instance[:, keep_indices]
    prediction = pipeline.named_steps['randomforestclassifier'].predict(X_instance)[0]
    predicted_class = le.inverse_transform([prediction])[0]
    return {"visualization": html, "predicted_class": predicted_class}
