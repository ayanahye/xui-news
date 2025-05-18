import pandas as pd

df_opendataval = pd.read_csv("data_valuation_results.csv")
df_betashap = pd.read_csv("leaderboard.csv")

df_merged = pd.merge(
    df_opendataval,
    df_betashap,
    on=["index", "text", "label"],
    suffixes=('_opendataval', '_betashap')
)

df_merged.to_csv("static/merged_leaderboard_results.csv", index=False)
