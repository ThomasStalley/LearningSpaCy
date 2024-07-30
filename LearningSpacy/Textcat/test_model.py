import spacy
from ml_datasets import imdb

# load in ml data:
train_data, valid_data = imdb()

# load in trained model:
nlp = spacy.load("output/model-best")

# test model on unseen case one:
text = train_data[522]
doc = nlp(text[0])
print("\n\ncase:", text)
print("doc.cats:", doc.cats)

# test model on unseen case two:
text = train_data[555]
doc = nlp(text[0])
print("\n\ncase:", text)
print("doc.cats:", doc.cats)

# test model on unseen case three:
text = train_data[588]
doc = nlp(text[0])
print("\n\ncase:", text)
print("doc.cats:", doc.cats)
