from flask import Flask, request, jsonify, send_from_directory, Response
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

        if method == "SHAP":
            result = shap_explainer.explain(combined_text, plot_type)
            if plot_type in ["beeswarm", "summary", "waterfall"]:
                return Response(result["visualization"], mimetype='text/html')
        elif method == "LIME":
            result = explain_with_lime(combined_text)
        else:
            result = {"error": "Invalid method"}
        
        response = jsonify(result)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    except Exception as e:
        print(f"API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
