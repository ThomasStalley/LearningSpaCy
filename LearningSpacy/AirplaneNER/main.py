import json

import spacy
from spacy.tokens import DocBin

# Load in training data:
with open("TrainingData/train.json") as json_data:
    training_data = json.load(json_data)

# Load in english pipeline model:
nlp = spacy.load("en_core_web_sm")

# Create a DocBin object:
db = DocBin()

# Convert json training data to .spacy training data:
for training_case in training_data:
    text = training_case["text"]
    doc = nlp.make_doc(text)

    ents = []
    entities = training_case["entities"]
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print(f"Skipping entity: {training_case}")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("TrainingData/train.spacy")

# Fill config file (base_config.cfg from spacy.org):
# $ python -m spacy init fill-config base_config.cfg config.cfg

# Training:
# $ python -m spacy train config.cfg
#                      --output ./OutputData --paths.train TrainingData/train.spacy --paths.dev TrainingData/train.spacy

nlp_best = spacy.load("OutputData/model-best")
doc = nlp_best("there was a flight named D16")
print("\ndoc:", doc)
for ent in doc.ents:
    print("-> ent.text:", ent.text)
    print("-> ent.label_:", ent.label_)
    print("-> ent.label:", ent.label)
    print("-> ent.start:", ent.start)
    print("-> ent.end:", ent.end)
    print("-> ent.start_char:", ent.start_char)
    print("-> ent.end_char:", ent.end_char)

nlp_best = spacy.load("OutputData/model-best")
doc = nlp_best("there was a made up plane called the A22 bomber!")
print("\ndoc:", doc)
for ent in doc.ents:
    print("-> ent.text:", ent.text)
    print("-> ent.label_:", ent.label_)
    print("-> ent.label:", ent.label)
    print("-> ent.start:", ent.start)
    print("-> ent.end:", ent.end)
    print("-> ent.start_char:", ent.start_char)
    print("-> ent.end_char:", ent.end_char)
