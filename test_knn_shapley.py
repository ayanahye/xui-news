import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from pydvl.valuation import Dataset
from pydvl.valuation.methods import KNNShapleyValuation

# reference: https://pydvl.org/stable/examples/shapley_knn_flowers/
# using knn-shapley to get a faster computation (can compute an exact value in almost linear time)

# load and sample data
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
df['combined_text'] = df['headline'] + " " + df['short_description'] + " " + df['authors'].fillna('')

categories = df['category'].unique()
selected_categories = pd.Series(categories).sample(n=3, random_state=42).tolist()
df_sampled = df[df['category'].isin(selected_categories)].sample(n=500, random_state=42)

X = df_sampled['combined_text'].reset_index(drop=True)
y = df_sampled['category'].reset_index(drop=True)

# convert category names to integers labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# convert to tfidf vectors
vectorizer = TfidfVectorizer(min_df=5)
X_train_vec = vectorizer.fit_transform(X_train_text).toarray()
X_test_vec = vectorizer.transform(X_test_text).toarray()

# debug
print("X_train_vec shape:", X_train_vec.shape)
print("y_train shape:", np.array(y_train).shape)
print("X_test_vec shape:", X_test_vec.shape)
print("y_test shape:", np.array(y_test).shape)

# gives each tfidf feature a name
# convert to dataset to make it work for knn-shapley
feature_names = [f"f{i}" for i in range(X_train_vec.shape[1])]
train_ds, _ = Dataset.from_arrays(X_train_vec, np.array(y_train), feature_names=feature_names)
test_ds, _ = Dataset.from_arrays(X_test_vec, np.array(y_test), feature_names=feature_names)

# debug
print("train_ds shape:", train_ds._x.shape)
print("train_ds y shape:", train_ds._y.shape)

# train the knn model
knn = KNeighborsClassifier(n_neighbors=5)
# measure how much each training point helps the models accuracy on the test set
valuation = KNNShapleyValuation(model=knn, test_data=test_ds, progress=True)

valuation.fit(train_ds)

# sort by shapley value
# following: https://pydvl.org/stable/examples/shapley_knn_flowers/
result = valuation.result.sort()

# debug
print("number of shapley values:", len(result.values))

print("top 20 most valuable training points: ")
for i in range(min(20, len(result.values))):
    idx = result.indices[i]
    print(f"Index {idx}: Value={result.values[i]:.5f} | Text='{X_train_text.iloc[idx][:60]}...' | Label={le.inverse_transform([y_train[idx]])[0]}")

print("\nbottom 20 least valuable training points: ")
for i in range(1, min(21, len(result.values)+1)):
    idx = result.indices[-i]
    print(f"Index {idx}: Value={result.values[-i]:.5f} | Text='{X_train_text.iloc[idx][:60]}...' | Label={le.inverse_transform([y_train[idx]])[0]}")
