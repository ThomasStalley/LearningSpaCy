import time

import numpy as np
import pandas as pd
import requests
import spacy
from spacy.language import Language
from spacy.matcher import Matcher

# ---------------------------------------------------------------------------------------------------------- Definitions
NLP = "...is the ability of a computer program to understand human language as it is spoken and written."
SPACY = "...is an open-source software library for advanced natural language processing."

# ---------------------------------------------------------------------------------------------------------- Sample Data
usa_text = open(file="Data/usa.txt", mode="r").read()
mlk_text = open(file="Data/mlk.txt", mode="r").read()
aiw_text = open(file="Data/aiw.txt", mode="r").read()
apple_text = open(file="Data/apple.txt", mode="r").read()
reuters_text = open(file="Data/reuters.txt", mode="r").read()

# ------------------------------------------------------------------------------------------------ Linguistic Annotation
nlp = spacy.load("en_core_web_sm")
usa_doc = nlp(usa_text)

print("\nFirst 10 tokens in usa doc:")
for token in usa_doc[:10]:
    print("->", token)

print("\nAll sentences in usa doc:")
for sent in usa_doc.sents:
    print("->", sent)

sentences = list(usa_doc.sents)
sentence1 = sentences[0]
token2 = sentence1[2]
print("\nsentence1:", sentence1)
print("token2 = sentence1[2] =", token2)
print("\ntoken2 attributes:")
print("->", "token2.text =", token2.text)
print("->", "token2.left_edge =", token2.left_edge)
print("->", "token2.right_edge =", token2.right_edge)
print("->", "token2.ent_type_ =", token2.ent_type_)
print("->", "token2.ent_iob_ =", token2.ent_iob_)
print("->", "token2.lemma_ =", token2.lemma_)
print("->", "token2.morph =", token2.morph)
print("->", "token2.pos_ =", token2.pos_)
print("->", "token2.dep_ =", token2.dep_)
print("->", "token2.lang_ =", token2.lang_)

mike_text = "Mike enjoys playing football!"
mike_doc = nlp(mike_text)
print(f"\nTokens in Mike doc:")
for token in mike_doc:
    print("->", token.text, token.pos_, token.dep_)

# --------------------------------------------------------------------------------------------- Named Entity Recognition
print(f"\nEntities in USA doc:")
for ent in usa_doc.ents:
    print("->", ent.text, ent.label_)

print(f"\nSentences in USA doc:")
for ent in usa_doc.sents:
    print("->", ent.text.strip())

# --------------------------------------------------------------------------------------------------------- Word Vectors
medium_nlp = spacy.load("en_core_web_md")
medium_usa_doc = nlp(usa_text)
sentence1 = list(medium_usa_doc.sents)[0]
token1 = list(medium_usa_doc)[0]
print("\nSentence1 from medium USA doc:")
print("->", sentence1)
print("\nToken1 from sentence1:")
print("->", token1)
print("\nToken1's word vector:")
print("->", token1.vector)

chosen_word = "country"
medium_nlp_strings = medium_nlp.vocab.strings
medium_nlp_vectors = medium_nlp.vocab.vectors
most_similar = medium_nlp_vectors.most_similar(np.asarray([medium_nlp_vectors[medium_nlp_strings[chosen_word]]]), n=10)
most_similar_words = [medium_nlp_strings[w] for w in most_similar[0][0]]

print("\nMost similar words to chosen word \"country\":")
for word in most_similar_words:
    print("->", word)

nlp = spacy.load("en_core_web_md")
french_fries_doc = nlp("I like salty fries and hamburgers.")
burgers_doc = nlp("Fast food tastes very good.")
french_fries_span = french_fries_doc[2:4]
burgers_token = french_fries_doc[5]

print("\nSimilarity using doc1.similarity(doc2) etc:")
print("->", french_fries_doc, "<->", burgers_doc, french_fries_doc.similarity(burgers_doc))
print("->", french_fries_span, "<->", burgers_token, french_fries_span.similarity(burgers_token))

# ------------------------------------------------------------------------------------------------------------ Pipelines
# Pipeline = Sequence of pipes that act on the input data, such as:
# (input: sample sentence) -> (entity ruler) -> (entity linker) -> (output=sentence with entities annotated)

process_shakespeare = False
if process_shakespeare:
    # Get shakespeare text:
    shakespeare_response = requests.get("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt")
    shakespeare_text = shakespeare_response.text

    # Create blank pipeline, add only a sentencizer step:
    blank_pipeline = spacy.blank("en")
    blank_pipeline.add_pipe("sentencizer")
    blank_pipeline.max_length = 5500000

    # Time how long it takes to create shakespeare doc:
    start_time = time.time()
    doc = blank_pipeline(shakespeare_text)
    end_time = time.time()

    # Show results:
    print("\nDoc of all shakespeare works:")
    print(f"-> Number of sentences: {len(list(doc.sents))}")
    print(f"-> Time taken to create doc using blank_pipeline: {end_time - start_time:.2f} seconds")

    # Analyze pipelines:
    print("\nUsing analyze pipes method:")
    print("-> blank_english_pipeline.analyze_pipes():", blank_pipeline.analyze_pipes())
    print("-> medium_english_pipeline.analyze_pipes():", medium_nlp.analyze_pipes())

# --------------------------------------------------------------------------------------------------------- Entity Ruler
# Toponym resolution = TR = resolution of things that could/do have multiple labels, that depend on context.
# IE Mr Deeds could be film title, or a name. Even in the same body of text.
# IE Paris could be a French city, a Texas city, a Kentucky city or even a persons name!

small_pipeline1 = spacy.load("en_core_web_sm")
sample_doc1 = small_pipeline1(sample_text := "Western Chestertenfieldville was referenced in Mr. Deeds.")

print(f"\nEntities in \"{sample_text}\" found by built in ner step:")
for ent in sample_doc1.ents:
    print("->", ent.text, ent.label_)

small_pipeline2 = spacy.load("en_core_web_sm")
ruler2 = small_pipeline2.add_pipe(factory_name="entity_ruler", before="ner")
patterns = [{"label": "GPE", "pattern": "Western Chestertenfieldville"}]
ruler2.add_patterns(patterns)

sample_doc2 = small_pipeline2(sample_text)
print(f"\nEntities in \"{sample_text}\" found by custom ruler step (1 pattern):")
for ent in sample_doc2.ents:
    print("->", ent.text, ent.label_)

small_pipeline3 = spacy.load("en_core_web_sm")
ruler3 = small_pipeline3.add_pipe(factory_name="entity_ruler", before="ner")
patterns = [{"label": "GPE", "pattern": "Western Chestertenfieldville"}, {"label": "FILM", "pattern": "Mr. Deeds"}]
ruler3.add_patterns(patterns)

sample_doc3 = small_pipeline3(sample_text)
print(f"\nEntities in \"{sample_text}\" found by custom ruler step (2 patterns):")
for ent in sample_doc3.ents:
    print("->", ent.text, ent.label_)

# -------------------------------------------------------------------------------------------------------------- Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
patterns = [[{"LIKE_EMAIL": True}]]
matcher.add(key="EMAIL_ADDRESS", patterns=patterns)

doc = nlp("This is an email address: main@exmaple.com")
matches = matcher(doc)
print("\nSample Text: \"This is an email address: main@exmaple.com\"")
print(f"LIKE_EMAIL matches:")
for match in matches:
    print("->", match)

nlp = spacy.load("en_core_web_sm")
doc = nlp(mlk_text)
matcher = Matcher(nlp.vocab)
patterns = [[{"POS": "PROPN"}]]
matcher.add(key="PROPER_NOUN", patterns=patterns)
matches = matcher(doc)

print("\nPROPER_NOUN matches[:10] in MLK text:")
for match in matches[:10]:
    print("->", doc[match[1]:match[2]])

nlp = spacy.load("en_core_web_sm")
doc = nlp(mlk_text)
matcher = Matcher(nlp.vocab)
patterns = [[{"POS": "PROPN", "OP": "+"}, {"POS": "VERB"}]]
matcher.add(key="PROPER_NOUN", patterns=patterns, greedy="LONGEST")

# Get matches and sort by start index:
matches = matcher(doc)
matches.sort(key=lambda x: x[1])

print("\nPROPER_NOUN + VERB (w/ operator=plus and greedy=longest) matches[:10] in MLK text:")
for match in matches[:10]:
    print("->", doc[match[1]:match[2]])

# Alice in wonderland text:
nlp = spacy.load("en_core_web_sm")
doc = nlp(aiw_text)
matcher = Matcher(nlp.vocab)

speak_lemmas = ["think", "say"]
patterns = [[
    {"ORTH": "'"},
    {"IS_ALPHA": True, "OP": "+"},
    {"IS_PUNCT": True, "OP": "*"},
    {"ORTH": "'"},
    {"POS": "VERB", "LEMMA": {"IN": speak_lemmas}, "OP": "?"},
    {"POS": "PROPN", "OP": "?"}
]]
matcher.add(key="SPEECH", patterns=patterns, greedy="LONGEST")
matches = matcher(doc)
matches.sort(key=lambda x: x[1])

print("\nSpeech pattern#1 matches from Alice in Wonderland text:")
for match in matches:
    print("->", doc[match[1]:match[2]])

speak_lemmas = ["think", "say"]
patterns = [[
    {"ORTH": "'"},
    {"IS_ALPHA": True, "OP": "+"},
    {"IS_PUNCT": True, "OP": "*"},
    {"ORTH": "'"},
    {"POS": "VERB", "LEMMA": {"IN": speak_lemmas}},
    {"POS": "PROPN", "OP": "+"},
    {"ORTH": "'"},
    {"IS_ALPHA": True, "OP": "+"},
    {"IS_PUNCT": True, "OP": "*"},
    {"ORTH": "'"},
]]
matcher.add(key="SPEECH", patterns=patterns, greedy="LONGEST")
matches = matcher(doc)
matches.sort(key=lambda x: x[1])

print("\nSpeech pattern#2 matches from Alice in Wonderland text:")
for match in matches:
    print("->", doc[match[1]:match[2]])

# ---------------------------------------------------------------------------------------------------- Custom Components
nlp = spacy.load("en_core_web_sm")
doc = nlp(sample_text := "Britain is a place. Mary is a person.")
print(f"\nSample text: \"{sample_text}\"")
print("Entities in sample text:")
for ent in doc.ents:
    print("->", ent.text, ent.label_)


# This component will be another pipe in the pipeline:
@Language.component("remove_gpe")
def remove_gpe(inputted_doc):
    entities = inputted_doc.ents
    for entity in entities:
        if entity.label == "GPE":
            entities.remove(ent)
    inputted_doc.ents = entities
    return inputted_doc


nlp.add_pipe("remove_gpe")
print(f"\nSample text: \"{sample_text}\"")
print("Entities in sample text (w/ custom component):")
for ent in doc.ents:
    print("->", ent.text, ent.label_)

# ------------------------------------------------------------------------------------------------- Financial Case Study
stocks_df = pd.read_csv("Data/stocks.tsv", sep="\t")
symbols = stocks_df.Symbol.tolist()
companies = stocks_df.CompanyName.tolist()

indexes_df = pd.read_csv("Data/indexes.tsv", sep="\t")
indexes = indexes_df.IndexName.tolist()
index_symbols = indexes_df.IndexSymbol.tolist()

exchanges_df = pd.read_csv("Data/exchanges.tsv", sep="\t")
exchanges = exchanges_df.ISOMIC.tolist() + exchanges_df["Google Prefix"].tolist()
descriptions = exchanges_df.Description.tolist()

patterns = []
stops = ["two"]
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Create stock patterns:
for symbol in symbols:
    patterns.append({"label": "STOCK", "pattern": symbol})
    for l in letters:
        patterns.append({"label": "STOCK", "pattern": symbol + f".{l}"})

# Create company patterns:
for company in companies:
    if company not in stops:
        patterns.append({"label": "COMPANY", "pattern": company})
        words = company.split()
        if len(words) > 1:
            new = " ".join(words[:2])
            patterns.append({"label": "COMPANY", "pattern": new})

# Create index patterns:
for index in indexes:
    patterns.append({"label": "INDEX", "pattern": index})
    versions = []
    words = index.split()
    caps = []
    for word in words:
        word = word.lower().capitalize()
        caps.append(word)
    versions.append(" ".join(caps))
    versions.append(words[0])
    versions.append(caps[0])
    versions.append(" ".join(caps[:2]))
    versions.append(" ".join(words[:2]))
    for version in versions:
        if version != "NYSE":
            patterns.append({"label": "INDEX", "pattern": version})
for symbol in index_symbols:
    patterns.append({"label": "INDEX", "pattern": symbol})

# Create exchange patterns:
for d in descriptions:
    patterns.append({"label": "STOCK_EXCHANGE", "pattern": d})
for e in exchanges:
    patterns.append({"label": "STOCK_EXCHANGE", "pattern": e})

financial_nlp = spacy.blank("en")
financial_ruler = financial_nlp.add_pipe("entity_ruler")
financial_ruler.add_patterns(patterns)

financial_doc = financial_nlp(reuters_text)
print("\nEntities in reuters article:")
for ent in financial_doc.ents:
    print("->", ent.text, ent.label_)

apple_doc = financial_nlp(apple_text)
print("\nEntities in apple computers text:")
for ent in apple_doc.ents:
    print("->", ent.text, ent.label_)
