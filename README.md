
# NLP Sentimental analysis using RoberTa and VADER
RoadMap
- importing amazon dataset
- data_transformation for Vader
- Model training
- Plotting results of VADER
- importing amazon dataset

- Model training for RoberTa
- Plotting results of RoberTa
- Comparitive Study



## Requirements

Install my-project with npm

```bash
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  import nltk
  nltk.download('vader_lexicon')
```
    
## Dataset Used: Amazon Reviews

![App Screenshot](https://snipboard.io/yKeBXc.jpg)





## Model - VADER

- return the 3 values- Pos, Neg and Compound
## Positive Score (pos):

- This score indicates the strength of positive sentiment in the text.
- It ranges from 0 to 1, where 1 indicates extreme positive      sentiment.

## Negative Score (neg):

- This score indicates the strength of negative sentiment in the text.
- It also ranges from 0 to 1, where 1 indicates extreme negative sentiment.
## Compound Score (compound):

- The compound score is a single metric that calculates the overall sentiment of the text.
- It's a normalized score that ranges from -1 (most negative) to +1 (most positive).
## User Text classified in numerical representation

![App Screenshot](https://snipboard.io/quOJ9V.jpg)


## VADER Predictions

- Compound score predicted by model and Score(user rating)

![App Screenshot](https://snipboard.io/Kv8LCQ.jpg)

To deploy this project run

```bash
fig, axs = plt.subplots(1,3 , figsize = [15,6])
sns.barplot(data = New_df, x = 'Score', y = 'compound', palette = 'hot', ax = axs[0])
axs[0].set_title("compound score of amazon reviews")
sns.barplot(data = New_df, x = 'Score', y = 'neg',palette = 'hot', ax = axs[1])
axs[1].set_title("neg score of amazon reviews")
sns.barplot(data = New_df, x = 'Score', y = 'pos',palette = 'hot', ax = axs[2])
axs[2].set_title("pos score of amazon reviews")
```
![App Screenshot](https://snipboard.io/dAY4Fi.jpg)


## RoberTa Predictions

- Pretrained model
- Trained on twitter base sentiments

```bash
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
```
- Results

![App Screenshot](https://snipboard.io/Vom2qL.jpg)


## Test Cases (RoberTa)

- When the Actual Score is 5 star but model predicted least value of pos
![App Screenshot](https://snipboard.io/z32Vyf.jpg)

- When the Actual Score is 5 star but model predicted least value of pos
![App Screenshot](https://snipboard.io/nAOJPS.jpg)

- When the Actual Score is 5 star but model predicted least value of pos

## Test Cases( VADER)
![App Screenshot](https://snipboard.io/GX9oQZ.jpg)

## Comparitive Study
![App Screenshot](https://snipboard.io/CMUpgw.jpg)