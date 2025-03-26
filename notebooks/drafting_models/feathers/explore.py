import pandas as pd

df = pd.read_feather(
    r"C:\Users\kkao\Documents\GitHub\OwlAboutFootball_S25\notebooks\kkao_models\feathers\draft_data2020.feather"
)

df.to_csv("draft_data2020.csv", index=False)
