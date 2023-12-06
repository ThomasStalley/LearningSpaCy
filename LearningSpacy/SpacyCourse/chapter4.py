import json

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span, DocBin

# Model Training:
# 1. Initialize the model weights randomly
# 2. Predict a few examples with the current weights
# 3. Compare prediction with true labels
# 4. Calculate how to change weights to improve predictions
# 5. Update weights slightly
# 6. Go back to 2.

# Key Definitions:
# -> Training data: Examples and their annotations.
# -> Text: The input text the model should predict a label for.
# -> Label: The label the model should predict.
# -> Gradient: How to change the weights.

# Process:
# Training data -> (Text + Label) -> Gradient -> Predictions with Model -> Updated Model

# Training data:
# -> Used to update the model.

# Evaluation/Development data:
# -> Data the model hasn't seen during training.
# -> Used to calculate how accurate the model is.
# -> Should be representative of the data the model will see at runtime.

# Can save data as binary files:
# -> train_docbin = DocBin(docs=train_docs)
# -> train_docbin.to_disk("./train.spacy")
# -> dev_docbin = DocBin(docs=dev_docs)
# -> dev_docbin.to_disk("./dev.spacy")

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 4.4")

with open("SampleData/iphone.json", encoding="utf8") as f:
    TEXTS = json.loads(f.read())

nlp = spacy.blank("en")
matcher = Matcher(nlp.vocab)

iphone_and_x_pattern = [{"LOWER": "iphone"}, {"LOWER": "x"}]
iphone_and_number_pattern = [{"LOWER": "iphone"}, {"IS_DIGIT": True}]
matcher.add(key="GADGET", patterns=[iphone_and_x_pattern, iphone_and_number_pattern])

docs = []
for doc in nlp.pipe(TEXTS):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label=match_id) for match_id, start, end in matches]
    print("\nText:", doc)
    print("Spans matched by custom matcher:", spans)
    doc.ents = spans
    docs.append(doc)

doc_bin = DocBin(docs=docs)
doc_bin.to_disk("TrainData/train.spacy")

# Training Config File:
# -> Single source of truth for all settings.
# -> Typically called config.cfg.
# -> Defines how to initialize the nlp object.
# -> Includes all settings about the pipeline components and their model implementations.
# -> Configures the training process and hyperparameters.
# -> Makes your training more reproducible.

# Model can unlearn old things, if overfitted on new data.
# e.g.: if you only update it with "WEBSITE", it can "unlearn" what a "PERSON" is
# Also known as "catastrophic forgetting" problem

# Label scheme needs to be consistent and not too specific
# -> For example: "CLOTHING" is better than "ADULT_CLOTHING" and "CHILDRENS_CLOTHING"
# -> Pick categories that are reflected in local context
# -> More generic is better than too specific
# -> Use rules to go from generic labels to specific categories

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 4.12")

nlp = spacy.blank("en")

doc1 = nlp("Reddit partners with Patreon to help creators build communities")
doc1.ents = [
    Span(doc1, 0, 1, label="WEBSITE"),
    Span(doc1, 3, 4, label="WEBSITE"),
]

doc2 = nlp("PewDiePie smashes YouTube record")
doc2.ents = [Span(doc2, 2, 3, label="WEBSITE")]

doc3 = nlp("Reddit founder Alexis Ohanian gave away two Metallica tickets to fans")
doc3.ents = [Span(doc3, 0, 1, label="WEBSITE")]

print("\nWebsite spans:")
print("\n-> Doc1:", doc1)
print("-> Doc1 ents:", doc1.ents)
print("\n-> Doc2:", doc2)
print("-> Doc2 ents:", doc2.ents)
print("\n-> Doc3:", doc3)
print("-> Doc3 ents:", doc3.ents)


nlp = spacy.blank("en")

doc1 = nlp("Reddit partners with Patreon to help creators build communities")
doc1.ents = [
    Span(doc1, 0, 1, label="WEBSITE"),
    Span(doc1, 3, 4, label="WEBSITE"),
]

doc2 = nlp("PewDiePie smashes YouTube record")
doc2.ents = [Span(doc2, 0, 1, label="PERSON"), Span(doc2, 2, 3, label="WEBSITE")]

doc3 = nlp("Reddit founder Alexis Ohanian gave away two Metallica tickets to fans")
doc3.ents = [Span(doc3, 0, 1, label="WEBSITE"), Span(doc3, 2, 4, label="PERSON")]

print("\nWebsite & Person spans:")
print("\n-> Doc1:", doc1)
print("-> Doc1 ents:", doc1.ents)
print("\n-> Doc2:", doc2)
print("-> Doc2 ents:", doc2.ents)
print("\n-> Doc3:", doc3)
print("-> Doc3 ents:", doc3.ents)