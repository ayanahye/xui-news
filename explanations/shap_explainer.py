import pandas as pd
import shap
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ShapExplainer:
    def __init__(self):
        self._last_params = None
        self._setup_model(
            min_df=5,
            remove_stopwords=True,
            ngram_range=(1,1),
            train_data_size=300,
            features_to_remove=[]
        )

    def _setup_model(
        self,
        min_df=5,
        remove_stopwords=True,
        ngram_range=(1,1),
        train_data_size=300,
        features_to_remove=None
    ):
        features_to_remove = features_to_remove or []
        params = (
            min_df,
            remove_stopwords,
            ngram_range,
            train_data_size,
            tuple(sorted(features_to_remove))
        )
        if self._last_params == params:
            return

        df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
        df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
        categories = df['category'].unique()
        selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
        df_sampled = df[df['category'].isin(selected_categories)]
        if train_data_size is not None:
            df_sampled = df_sampled.sample(n=train_data_size, random_state=42)

        X = df_sampled['combined_text']
        y = df_sampled['category']

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

        self.vectorizer = vectorizer
        self.feature_names = feature_names

        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        self.X_train = X_train
        self.X_test = X_test
        self.X = X

        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        self.explainer = shap.Explainer(self.model, X_train, feature_names=self.feature_names)
        self._last_params = params

    def get_feature_names(self):
        return list(self.feature_names)

    def predict_class(
        self,
        text,
        min_df=5,
        remove_stopwords=True,
        ngram_range=(1,1),
        train_data_size=300,
        features_to_remove=None
    ):
        self._setup_model(min_df, remove_stopwords, ngram_range, train_data_size, features_to_remove)
        X_instance = self.vectorizer.transform([text]).toarray()

        if features_to_remove:
            keep_indices = [i for i, f in enumerate(self.vectorizer.get_feature_names_out()) if f not in features_to_remove]
            X_instance = X_instance[:, keep_indices]
        prediction = self.model.predict(X_instance)[0]
        predicted_class = self.le.inverse_transform([prediction])[0]
        return predicted_class

    def explain(
        self,
        text,
        plot_type="force",
        min_df=5,
        top_n=15,
        remove_stopwords=True,
        ngram_range=(1,1),
        train_data_size=300,
        features_to_remove=None
    ):
        self._setup_model(min_df, remove_stopwords, ngram_range, train_data_size, features_to_remove)
        X_instance = self.vectorizer.transform([text]).toarray()
        if features_to_remove:
            keep_indices = [i for i, f in enumerate(self.vectorizer.get_feature_names_out()) if f not in features_to_remove]
            X_instance = X_instance[:, keep_indices]
        prediction = self.model.predict(X_instance)[0]
        predicted_class = self.le.inverse_transform([prediction])[0]
        shap_values = self.explainer(X_instance)
        global_shap_values = self.explainer(self.X_test)
        html = None

        def get_shap_and_base(shap_values, sample_idx, class_idx):
            if len(shap_values.values.shape) == 3:
                shap_arr = shap_values.values[sample_idx, :, class_idx]
                base_val = shap_values.base_values[sample_idx, class_idx]
            elif len(shap_values.values.shape) == 2:
                shap_arr = shap_values.values[sample_idx, :]
                base_val = shap_values.base_values[sample_idx]
            else:
                raise ValueError("SH values shape: {}".format(shap_values.values.shape))
            return shap_arr, base_val

        def get_top_features(features, shap_vals, feature_names, top_n=15):
            top_idx = np.argsort(np.abs(shap_vals))[-top_n:]
            filtered_shap = np.zeros_like(shap_vals)
            filtered_features = np.zeros_like(features)
            filtered_shap[top_idx] = shap_vals[top_idx]
            filtered_features[top_idx] = features[top_idx]
            return pd.Series(filtered_features, index=feature_names), filtered_shap

        def beeswarm_plotly(shap_values, feature_names, class_idx=0, max_display=20):
            if len(shap_values.values.shape) == 3:
                mean_abs_shap = np.abs(shap_values.values[:,:,class_idx]).mean(axis=0)
                vals = shap_values.values[:,:,class_idx]
            else:
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                vals = shap_values.values
            top_idx = np.argsort(mean_abs_shap)[-max_display:]
            df = pd.DataFrame({
                "feature": np.repeat([feature_names[i] for i in top_idx], vals.shape[0]),
                "shap_value": vals[:,top_idx].flatten()
            })
            fig = px.strip(df, x="shap_value", y="feature", orientation="h", title="Beeswarm Plot (Interactive)")
            fig.update_layout(height=600)
            return fig.to_html(full_html=False, include_plotlyjs='cdn')

        def summary_plotly(shap_values, feature_names, class_idx=0, max_display=20):
            if len(shap_values.values.shape) == 3:
                mean_abs_shap = np.abs(shap_values.values[:,:,class_idx]).mean(axis=0)
            else:
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            top_idx = np.argsort(mean_abs_shap)[-max_display:]
            df = pd.DataFrame({
                "feature": [feature_names[i] for i in top_idx],
                "mean_abs_shap": mean_abs_shap[top_idx]
            })
            fig = px.bar(df, x="mean_abs_shap", y="feature", orientation="h", title="Summary Plot (Interactive)")
            fig.update_layout(height=600)
            return fig.to_html(full_html=False, include_plotlyjs='cdn')

        def waterfall_plotly(shap_values, feature_names, ind=0, class_idx=0, max_display=10):
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

        if plot_type == "force":
            shap_arr, base_val = get_shap_and_base(shap_values, 0, prediction)
            filtered_features, filtered_shap = get_top_features(
                X_instance[0], shap_arr, self.feature_names, top_n=top_n
            )
            html = shap.plots.force(
                base_value=base_val,
                shap_values=filtered_shap,
                features=filtered_features,
                feature_names=self.feature_names,
                matplotlib=False
            )
            html = f"<head>{shap.getjs()}</head><body>{html.html()}</body>"
        elif plot_type == "beeswarm":
            html = beeswarm_plotly(global_shap_values, self.feature_names, class_idx=prediction, max_display=top_n)
        elif plot_type == "summary":
            html = summary_plotly(global_shap_values, self.feature_names, class_idx=prediction, max_display=top_n)
        elif plot_type == "waterfall":
            html = waterfall_plotly(shap_values, self.feature_names, ind=0, class_idx=prediction, max_display=top_n)
        else:
            html = "<p>Ppot type not supported.</p>"

        return {"visualization": html, "predicted_class": predicted_class}
