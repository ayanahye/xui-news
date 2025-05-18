import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from opendataval.dataloader import DataFetcher
from opendataval.dataval import DVRL, DataOob
from opendataval.experiment import ExperimentMediator
from opendataval.model import ModelFactory

# just chose functions i thought were needed. but note to self to look over opendataval library again because not certain i chose the right / complete functions
# note to self -- i added log loss not sure its permissible or not but keeping consistent w beta shapley implementation im using

df_selected = pd.read_csv("data_used/sampled_data_used.csv").reset_index(drop=True)
splits = np.load("data_used/split_indices.npz")
idx_train = splits['idx_train']
idx_val = splits['idx_val']
idx_test = splits['idx_test']

X = df_selected['combined_text']
y = df_selected['category']

vectorizer = TfidfVectorizer(min_df=5)
X_vec = vectorizer.fit_transform(X).toarray()
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train = X_vec[idx_train]
y_train = y_encoded[idx_train]
X_valid = X_vec[idx_val]
y_valid = y_encoded[idx_val]
X_test = X_vec[idx_test]
y_test = y_encoded[idx_test]

fetcher = DataFetcher.from_data_splits(
    x_train=X_train,
    y_train=y_train,
    x_valid=X_valid,
    y_valid=y_valid,
    x_test=X_test,
    y_test=y_test,
    one_hot=False
)

model = ModelFactory("logisticregression", fetcher=fetcher)

exper_med = ExperimentMediator(
    fetcher=fetcher,
    pred_model=model,
    train_kwargs={},
    metric_name="log_loss"
)

data_evaluators = [DVRL(rl_epochs=2000), DataOob(num_models=1000)]

exper_med.compute_data_values(data_evaluators)

results = {}
for evaluator in exper_med.data_evaluators:
    method_name = evaluator.__class__.__name__
    try:
        data_values = evaluator.evaluate_data_values()
        results[method_name] = data_values
        print(f"{method_name}: Data values shape {data_values.shape}")
    except Exception as e:
        print(f"no data values for {method_name}: {e}")

df_results = pd.DataFrame(results)
df_results['text'] = df_selected.loc[idx_train, 'combined_text'].values
df_results['label'] = df_selected.loc[idx_train, 'category'].values
df_results['index'] = idx_train
df_results.to_csv("data_valuation_results.csv", index=False)