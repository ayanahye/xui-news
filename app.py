from flask import Flask, request, jsonify, send_from_directory, Response
import pandas as pd
from explanations.shap_explainer import ShapExplainer
from explanations.lime_explainer import explain_with_lime
import os

app = Flask(__name__)

shap_explainer = ShapExplainer()

@app.route('/')
def index():
    return send_from_directory('.', 'templates/index.html')

@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        combined_text = data['combined_text']
        method = data['method']
        plot_type = data.get('plot_type', 'force')
        min_df = int(data.get('min_df', 5))
        top_n = int(data.get('top_n', 15))
        # feature engineering new params
        remove_stopwords = data.get('remove_stopwords', True)
        ngram_range = data.get('ngram_range', [1, 1])
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)


        train_data_size = data.get('train_data_size', 500)
        if train_data_size == "all":
            train_data_size = 5000
        else:
            train_data_size = int(train_data_size)

        features_to_remove = data.get('features_to_remove', [])

        if method == "SHAP":
            result = shap_explainer.explain(
                combined_text,
                plot_type=plot_type,
                min_df=min_df,
                top_n=top_n,
                remove_stopwords=remove_stopwords,
                ngram_range=ngram_range,
                train_data_size=train_data_size,
                features_to_remove=features_to_remove
            )
            if plot_type in ["beeswarm", "summary", "waterfall"]:
                return Response(result["visualization"], mimetype='text/html')
        elif method == "LIME":
            result = explain_with_lime(
                combined_text,
                min_df=min_df,
                num_features=top_n,
                remove_stopwords=remove_stopwords,
                ngram_range=ngram_range,
                train_data_size=train_data_size,
                features_to_remove=features_to_remove
            )
        else:
            result = {"error": "Invalid method"}

        response = jsonify(result)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    except Exception as e:
        print(f"API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        combined_text = data['combined_text']
        min_df = int(data.get('min_df', 5))
        remove_stopwords = data.get('remove_stopwords', True)
        ngram_range = data.get('ngram_range', [1, 1])
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        train_data_size = data.get('train_data_size', 500)
        if train_data_size == "all":
            train_data_size = 5000
        else:
            train_data_size = int(train_data_size)

        features_to_remove = data.get('features_to_remove', [])

        predicted_class = shap_explainer.predict_class(
            combined_text,
            min_df=min_df,
            remove_stopwords=remove_stopwords,
            ngram_range=ngram_range,
            train_data_size=train_data_size,
            features_to_remove=features_to_remove
        )
        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        print(f"Prediction API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/features')
def features():
    try:
        features = list(shap_explainer.get_feature_names())
        return jsonify(features)
    except Exception as e:
        return jsonify([])

@app.route('/sampled_articles')
def sampled_articles():
    df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
    df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
    categories = df['category'].unique()
    selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
    df_sampled = df[df['category'].isin(selected_categories)].sample(n=500, random_state=42)
    articles = []
    per_category = 3
    for cat in selected_categories:
        cat_articles = df_sampled[df_sampled['category'] == cat].sample(n=per_category, random_state=1)
        articles.extend(cat_articles[['headline', 'short_description', 'authors', 'category']].to_dict(orient='records'))
    return jsonify(articles)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
