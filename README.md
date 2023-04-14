## Introduction

This Github repository contains a Python code created by Oğuzhan Öztürk. The code uses the Top2Vec library to generate topic models and create word clouds for a given input CSV file containing review data.

## Installation

To use this code, first install the required libraries:

```python
!pip install top2vec
!pip install top2vec[sentence_encoders]
!pip install top2vec[sentence_transformers]
!pip install top2vec[indexing]
!pip install numpy==1.23.5
```

In addition, you will need to import pandas, nltk, and NLTK resources. To download the necessary resources from NLTK, use the following commands:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

After installing the necessary libraries, you can run the code by importing it and specifying the path of your input CSV file:

```python
import pandas as pd 

reviews = pd.read_csv('CSV FILE PATH', on_bad_lines='skip')
```

The next step is to clean the review text by running the `clean_text` function:

```python
def clean_text(review):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(review)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text
  
reviews['review']=reviews['review'].apply(clean_text)
```

Next, generate the topic models by running the following lines of code:

```python
from top2vec import Top2Vec

model = Top2Vec(list(reviews['review'].to_numpy()), embedding_model='universal-sentence-encoder-large',use_embedding_model_tokenizer=True,split_documents=True)
model.get_num_topics()
topic_sizes, topic_nums = model.get_topic_sizes()
```

Finally, create the topic word clouds by running the following loop:

```python
for topic in topic_nums:
  model.generate_topic_wordcloud(topic)
```

## Author

This code was created by Oğuzhan Öztürk. For any inquiries or suggestions, please contact the author at oguzhanozturk0@outlook.com.
