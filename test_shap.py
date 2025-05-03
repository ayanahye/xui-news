import pandas as pd
import numpy as np
import shap
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')
print(df.iloc[5])

# we sample 500 points (400 for training) from 3 categories
categories = df['category'].unique()
selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
print(selected_categories)
df_sampled = df[df['category'].isin(selected_categories)].sample(n=500, random_state=42)

X = df_sampled['combined_text']
print(X.iloc[5])
y = df_sampled['category']

# word must appear in at least 5 documents to become a feature
vectorizer = TfidfVectorizer(min_df=5)
X_vec = vectorizer.fit_transform(X).toarray()
# 500 samples, 480 features 
print(X_vec.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
print(f"\ntotal train data points: {len(X_train)}")

model = RandomForestClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.inverse_transform(np.unique(y_test))))

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print(f"Overall model accuracy: {acc:.2f}")

feature_names = vectorizer.get_feature_names_out()
# create explainer object w model and training data so we can retrain if needed
explainer = shap.Explainer(model, X_train, feature_names=feature_names)
# compute shap values using test set
shap_values = explainer(X_test)

print(f"\nTotal test data points: {len(X_test)}")

shap.initjs()
ind = 5
true_label = le.inverse_transform([y_test[ind]])[0]
predicted_label = le.inverse_transform([y_pred[ind]])[0]
is_correct = y_pred[ind] == y_test[ind]

X_train_text, X_test_text = train_test_split(X, test_size=0.2, stratify=y_encoded, random_state=42)

print(f"\nExplanation for index: {ind}")
print(f"Original text: {X_test_text.iloc[ind]}")
print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")
print(f"Prediction {'correct' if is_correct else 'incorrect'}")

# shap values of features for each prediction
# increment/decrement for each word to go from average prediction to the specific prediction for this sample
# avg prediction is the E(F(X)) over the test set, while f(x) is our prediction for this sample

# single instance based
shap.plots.waterfall(shap_values[ind, :, y_pred[ind]], max_display=20)

# can also plot for a specific feature/word
# each point is the feature in a diff sample of the test set 
# x is the tf-idf value and y is the shap value
if "news" in feature_names:
    feature_index = feature_names.tolist().index("news")
    # plot for feature in class 1
    shap.plots.scatter(shap_values[:, feature_index, 1])
else:
    print('"news" not found in the feature names.')

# force plot shows the base value on the far left which is the expected value for the models prediction over training data
# then each feature is coded where red bars push it higher to a specific class and blue bars push it lower

# single instances based
shap_html = shap.plots.force(shap_values[ind, :, y_pred[ind]], matplotlib=False)
shap.save_html("force_plot.html", shap_html)

#shap.plots.beeswarm(shap_values)

# plots shap values for every feature for every sample
# 1 row shows how that feature affects predictions across all samples
# features at the top are most important (sorts feature by sum of shap value magnitudes)
# horizontal axis shows how much the shap values vary for each feature
# tells how much influence the feature can have in either direction 
# color tells u if its high (red) or low (blue)

# these are global
# which features were most important overall for that class
shap.plots.beeswarm(shap_values[:, :, 0], max_display=50)  # For class 0
shap.plots.beeswarm(shap_values[:, :, 1], max_display=50)  # For class 1
shap.plots.beeswarm(shap_values[:, :, 2], max_display=50)  # For class 2

# this is impact of each feature globally

shap.summary_plot(shap_values, X_test, show=True, max_display=50)

print(X_vec.shape[1])