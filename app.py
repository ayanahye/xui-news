from flask import Flask, request, jsonify, send_from_directory, Response
import pandas as pd
from explanations.shap_explainer import ShapExplainer
from explanations.lime_explainer import explain_with_lime
import os
from explanations.knn_shap_explainer import KNNShapExplainer

from explanations.beta_shap_explainer import BetaShapExplainer


app = Flask(__name__)

shap_explainer = ShapExplainer()
knn_shap_explainer = KNNShapExplainer()
beta_shap_explainer = BetaShapExplainer()

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


        train_data_size = data.get('train_data_size', 100)
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
        train_data_size = data.get('train_data_size', 100)
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
    df_sampled = df[df['category'].isin(selected_categories)].sample(n=100, random_state=42)
    articles = []
    per_category = 3
    for cat in selected_categories:
        cat_df = df_sampled[df_sampled['category'] == cat]
        sample_size = min(per_category, len(cat_df)) 
        if sample_size == 0:
            continue  
        cat_articles = cat_df.sample(n=sample_size, random_state=1)
        articles.extend(cat_articles[['headline', 'short_description', 'authors', 'category']].to_dict(orient='records'))

    return jsonify(articles)

@app.route('/leaderboard')
def leaderboard():
    try:
        knn_leaderboard = knn_shap_explainer.compute_leaderboard()
        beta_leaderboard = beta_shap_explainer.compute_leaderboard()

        beta_dict = {item['index']: item for item in beta_leaderboard}
        combined = []
        for knn_item in knn_leaderboard:
            idx = knn_item['index']
            beta_item = beta_dict.get(idx, {})
            combined.append({
                "index": idx,
                "value": knn_item['value'], 
                "beta_shapley": beta_item.get('beta_shapley', None),
                "loo": beta_item.get('loo', None),
                "text": knn_item['text'],
                "label": knn_item['label']
            })
        combined = sorted(combined, key=lambda x: x['value'], reverse=True)
        return jsonify(combined)
    except Exception as e:
        print(f"Leaderboard API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
