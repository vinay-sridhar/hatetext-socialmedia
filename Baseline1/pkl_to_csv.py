import pandas as pd

df = pd.read_csv('data/hate_norm_with_span.csv')

df.to_pickle('data/hate_norm_combined.pkl')
