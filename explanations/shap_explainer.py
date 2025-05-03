import pandas as pd
import shap
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

def beeswarm_plotly(shap_values, feature_names, class_idx=0, max_display=20):
    mean_abs_shap = np.abs(shap_values.values[:,:,class_idx]).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-max_display:]
    df = pd.DataFrame({
        "feature": np.repeat([feature_names[i] for i in top_idx], shap_values.values.shape[0]),
        "shap_value": shap_values.values[:,top_idx,class_idx].flatten()
    })
    fig = px.strip(df, x="shap_value", y="feature", orientation="h", title="Beeswarm Plot (Interactive)")
    fig.update_layout(height=600)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def summary_plotly(shap_values, feature_names, class_idx=0, max_display=20):
    mean_abs_shap = np.abs(shap_values.values[:,:,class_idx]).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-max_display:]
    df = pd.DataFrame({
        "feature": [feature_names[i] for i in top_idx],
        "mean_abs_shap": mean_abs_shap[top_idx]
    })
    fig = px.bar(df, x="mean_abs_shap", y="feature", orientation="h", title="Summary Plot (Interactive)")
    fig.update_layout(height=600)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def waterfall_plotly(shap_values, feature_names, ind=0, class_idx=0, max_display=10):
    import plotly.graph_objects as go
    if len(shap_values.values.shape) == 3:
        values = shap_values.values[ind, :, class_idx]
        base_value = shap_values.base_values[ind, class_idx]
    elif len(shap_values.values.shape) == 2:
        values = shap_values.values[ind, :]
        base_value = shap_values.base_values[ind]
    else:
        return "<div>ugh SHAP values shape</div>"

    if np.all(values == 0):
        return "<div>no SHAP values to plot</div>"

    features = feature_names
    top_idx = np.argsort(np.abs(values))[-max_display:]
    features = features[top_idx]
    values = values[top_idx]
    x = ['Base value'] + list(features) + ['Model output']
    y = [base_value] + list(values) + [base_value + np.sum(values)]
    fig = go.Figure(go.Waterfall(
        x=x,
        y=y,
        measure=['absolute'] + ['relative']*len(values) + ['total'],
        textposition="outside",
        text=[f"{v:.2f}" for v in y],
        connector={"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="SHAP Waterfall Plot (Interactive)", showlegend=False, height=500)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

class ShapExplainer:
    def explain(self, text, plot_type="force"):
        X_instance = vectorizer.transform([text]).toarray()
        prediction = model.predict(X_instance)[0]
        predicted_class = le.inverse_transform([prediction])[0]
        shap_values = explainer(X_instance)
        global_shap_values = explainer(X_test)
        html = None

        if plot_type == "force":
            # force is not working... feature names all clumped to the side
            features_df = pd.DataFrame([X_instance[0]], columns=feature_names)
            print(feature_names)
            html = shap.plots.force(
                shap_values[0, :, prediction],
                features=features_df.iloc[0],
                feature_names=feature_names,
                matplotlib=False
            )
            html = f"<head>{shap.getjs()}</head><body>{html.html()}</body>"
        elif plot_type == "beeswarm":
            html = beeswarm_plotly(global_shap_values, feature_names, class_idx=prediction)
        elif plot_type == "summary":
            html = summary_plotly(global_shap_values, feature_names, class_idx=prediction)
        elif plot_type == "waterfall":
            html = waterfall_plotly(shap_values, feature_names, ind=0, class_idx=prediction)
        else:
            html = "<p>Ppot type not supported.</p>"

        return {"visualization": html, "predicted_class": predicted_class}
