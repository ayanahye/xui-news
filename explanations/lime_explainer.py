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

def get_lime_pipeline(min_df=5, remove_stopwords=True, ngram_range=(1,1), train_data_size=500):
    cache_key = (min_df, remove_stopwords, ngram_range, train_data_size)
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
    pipeline = make_pipeline(vectorizer, RandomForestClassifier())
    pipeline.fit(X, y_encoded)
    _pipeline_cache[cache_key] = (pipeline, vectorizer, le, class_names)
    return pipeline, vectorizer, le, class_names

def explain_with_lime(
    text_instance,
    min_df=5,
    num_features=6,
    remove_stopwords=True,
    ngram_range=(1,1),
    train_data_size=500
):
    pipeline, vectorizer, le, class_names = get_lime_pipeline(min_df, remove_stopwords, ngram_range, train_data_size)
    explainer = LimeTextExplainer(class_names=class_names)
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
