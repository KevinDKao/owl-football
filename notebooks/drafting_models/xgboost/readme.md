# Understanding Draft Status in the Dataset

## Identifying Undrafted Players

In the `draft_data.csv` file, the draft status of players is represented by several key columns:

- `overall`: The overall pick number in the draft
- `round`: The draft round number
- `pick`: The pick number within the round

**Important**: When any of these values is 0, it indicates that the player was not drafted. This is a critical distinction in the dataset that allows us to identify which players made it to the NFL draft and which did not.

## Creating the Target Variable

For our XGBoost prediction model, we created a binary target variable called `drafted` using this logic:
```python
df['drafted'] = (df['overall'] > 0).astype(int)
```

This transforms the data into:
- `drafted = 1`: Player was selected in the NFL draft (overall pick number > 0)
- `drafted = 0`: Player went undrafted (overall pick number = 0)

## Significance of Undrafted Status

The inclusion of undrafted players in our dataset is crucial for several reasons:

1. **Complete Picture**: It provides a complete picture of the college-to-NFL pipeline, not just the success stories.

2. **Balanced Learning**: It allows our model to learn the differences between drafted and undrafted players, creating a more balanced and realistic prediction system.

3. **Threshold Analysis**: It enables us to analyze the threshold between being drafted and undrafted - what minimal statistics or attributes might push a player over the edge into draft consideration.

4. **Free Agent Potential**: Many undrafted players still sign with NFL teams as undrafted free agents and have successful careers. The model might identify patterns among these players who were overlooked in the draft but had NFL-caliber skills.

## Model Applications for Undrafted Players

Our model doesn't just predict who will be drafted, but can also:

1. **Identify Near-Miss Prospects**: Players who narrowly missed being drafted according to our model might be prime candidates for undrafted free agent signings.

2. **Development Opportunities**: Highlight specific areas where undrafted players fell short, providing targeted development opportunities for players hoping to enter the NFL through free agency.

3. **Draft Strategy Insights**: Help teams identify potential late-round steals or priority free agent targets who have similar profiles to drafted players.

4. **Career Path Guidance**: Assist college players in making informed decisions about declaring for the draft versus returning to school based on their predicted draft probability.

By including both drafted and undrafted players in our analysis, the model provides a comprehensive view of what factors truly influence NFL draft selection, helping all stakeholders make more informed decisions throughout the draft process.
