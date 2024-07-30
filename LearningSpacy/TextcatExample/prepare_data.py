import spacy
from ml_datasets import imdb
from spacy.tokens import DocBin

nlp = spacy.load("en_core_web_sm")


def make_docs(data, name):
    print(f"processing {name}...", end="")
    docs = []
    for doc, label in nlp.pipe(data, as_tuples=True):
        if label == "neg":
            doc.cats["positive"] = 0
            doc.cats["negative"] = 1
        else:
            doc.cats["positive"] = 1
            doc.cats["negative"] = 0
        docs.append(doc)
    print("completed!")
    return docs


# load in ml data:
train_data, valid_data = imdb()

# transform the training data, and save as binary to disk:
train_docs = make_docs(data=train_data[:500], name="train")
doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk("./data/train.spacy")

# transform the validation data, and save as binary to disk:
valid_docs = make_docs(data=valid_data[:500], name="valid")
doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk("./data/valid.spacy")
