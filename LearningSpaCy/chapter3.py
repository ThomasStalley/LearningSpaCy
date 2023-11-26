import json

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Token, Span, Doc

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.1")

# NLP Pipeline:
# -> First, the tokenizer is applied to turn the string of text into a Doc object.
# -> Next, a series of pipeline components is applied to the doc in order.
# -> i.e. the tagger, then the parser, then the entity recognizer.
# -> Finally, the processed doc is returned, so you can work with it.

# Built-in pipeline components:
# tagger 	Part-of-speech tagger 	   Token.tag, Token.pos
# parser 	Dependency parser 	       Token.dep, Token.head, Doc.sents, Doc.noun_chunks
# ner 	    Named entity recognizer    Doc.ents, Token.ent_iob, Token.ent_type
# textcat 	Text classifier 	       Doc.cats

# -> The part-of-speech tagger sets the token.tag and token.pos attributes.
# -> The dependency parser adds the token.dep and token.head attributes and
#    is also responsible for detecting sentences and base noun phrases, also known as noun chunks.
# -> The named entity recognizer adds the detected entities to the doc.ents
#    property. It also sets entity type attributes on the tokens that indicate if a token is part of an entity or not.
# -> Finally, the text classifier sets category labels that apply to the whole text, and adds them to doc.cats.

# The text classifier is not included in any of the trained pipelines by default.
# You can use the text classifier to train your own system.

nlp = spacy.load("en_core_web_sm")

print("\nnlp:", nlp)

print("\nnlp.pipe_names:")
for ele in nlp.pipe_names:
    print("->", ele)

print("\nnlp.pipeline:")
for ele in nlp.pipeline:
    print("->", ele)

# What does spaCy do when you call nlp on a string of text?
# -> Tokenize the text and apply each pipeline component in order.

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.4")


# Custom components are executed automatically when you call the nlp object on a text.
# They're especially useful for adding your own custom metadata to documents and tokens.
# You can also use them to update built-in attributes, like the named entity spans.


@Language.component("custom_component")
def custom_component_function(doc):
    print("-> Doc length:", len(doc))
    return doc


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("custom_component", first=True)
print("\nNLP Pipenames:", nlp.pipe_names)

print("\nnlp(\"Hello world!\"):")
doc = nlp("Hello world!")

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.4")


@Language.component("length_component")
def length_component_function(doc):
    doc_length = len(doc)
    print(f"-> This document is {doc_length} tokens long.")
    return doc


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("length_component", first=True)
print("\nNLP Pipenames:", nlp.pipe_names)

print("\nnlp(\"This is a sentence.\"):")
doc = nlp("This is a sentence.")

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.7")


@Language.component("animal_component")
def animal_component_function(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]
    doc.ents = spans
    return doc


nlp = spacy.load("en_core_web_sm")

animal_patterns_list = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]
animal_patterns = list(nlp.pipe(animal_patterns_list))
matcher = PhraseMatcher(nlp.vocab)
matcher.add("ANIMAL", animal_patterns)
print("\nanimal_patterns:", animal_patterns)

nlp.add_pipe("animal_component", after="ner")
print("\nNLP Pipenames:", nlp.pipe_names)

sample_text = "I have a cat and a Golden Retriever"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

print("\nEntities (text & label) overwritten by custom component to match chosen animals:")
for ent in doc.ents:
    print("->", ent.text, ent.label_)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.8")


# Custom attributes let you add any metadata to docs, tokens and spans.
# Custom attributes are available via the ._ (dot underscore) property.

# Custom attribute types:
# -> Attribute extensions
# -> Property extensions
# -> Method extensions

# Set extensions on the Doc, Token and Span:
# -> Doc.set_extension("title", default=None)
# -> Token.set_extension("is_color", default=False)
# -> Span.set_extension("has_color", default=False)


def get_is_color(token):
    colors = ["red", "yellow", "blue"]
    return token.text in colors


def get_has_color(span):
    colors = ["red", "yellow", "blue"]
    return any(token.text in colors for token in span)


def has_token(doc, token_text):
    in_doc = token_text in [token.text for token in doc]
    return in_doc


Token.set_extension("is_color", getter=get_is_color)
Span.set_extension("has_color", getter=get_has_color)
Doc.set_extension("has_token", method=has_token)

sample_text = "The sky is blue."
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)
token = doc[3]

print(f"\nToken: \"{token}\"")
print(f"-> token._.is_color: {token._.is_color}")
print(f"-> token.text: {token.text}")

span = doc[1:4]
print(f"\nSpan: \"{span}\"")
print(f"-> token._.has_color: {span._.has_color}")
print(f"-> token.text: {span.text}")

span = doc[0:2]
print(f"\nSpan: \"{span}\"")
print(f"-> span._.has_color: {span._.has_color}")
print(f"-> span.text: {span.text}")

print(f"\nDoc: \"{doc}\"")
print(f"-> doc._.has_token(\"blue\"): {doc._.has_token('blue')}")
print(f"-> doc._.has_token(\"cloud\"): {doc._.has_token('cloud')}")

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.9")

Token.set_extension("is_country", default=False)

sample_text = "I live in Spain."
print(f"\nSample text: \"{sample_text}\"")
nlp = spacy.blank("en")
doc = nlp(sample_text)

token = doc[3]
token._.is_country = True
print("\n(token._.is_country, token.text) for token in sample_text:")
for token in doc:
    print("->", (token.text, token._.is_country))


def get_reversed(token):
    return token.text[::-1]


Token.set_extension("reversed", getter=get_reversed)

sample_text = "All generalizations are false, including this one."
nlp = spacy.blank("en")
doc = nlp(sample_text)

print(f"\nSample text: \"{sample_text}\"")
print("\nUsing token extension \"reversed\":")
for token in doc:
    print("-> Reversed:", token._.reversed)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.9")


def get_has_number(doc):
    return any(token.like_num for token in doc)


Doc.set_extension("has_number", getter=get_has_number)

sample_text = "The museum closed for five years in 2012."
nlp = spacy.blank("en")
doc = nlp(sample_text)

print(f"\nSample text: \"{sample_text}\"")
print("-> has_number:", doc._.has_number)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.10")


def to_html(span, tag):
    return f"<{tag}>{span.text}</{tag}>"


Span.set_extension("to_html", method=to_html)

nlp = spacy.blank("en")
sample_text = "Hello world, this is a sentence."
doc = nlp(sample_text)
span = doc[0:2]

print(f"\nSample text: \"{sample_text}\"")
print(f"\nSpan: {span}")
print(f"\nUsing span extension \"to_html\":")
print("->", span._.to_html("strong"))

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.11")


def get_wikipedia_url(span):
    if span.label_ in ("PERSON", "ORG", "GPE", "LOCATION"):
        entity_text = span.text.replace(" ", "_")
        return "https://en.wikipedia.org/w/index.php?search=" + entity_text


Span.set_extension("wikipedia_url", getter=get_wikipedia_url)

sample_text = (
    "In over fifty years from his very first recordings right through to his "
    "last album, David Bowie was at the vanguard of contemporary culture."
)
nlp = spacy.load("en_core_web_sm")
doc = nlp(sample_text)

print(f"\nSample text: \"{sample_text}\"")
print(f"\nDoc entities (ent.text, ent._.wikipedia_url):")
for ent in doc.ents:
    print("->", ent.text, ent._.wikipedia_url)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.12")

with open("SampleData/countries.json", encoding="utf8") as f:
    COUNTRIES = json.loads(f.read())

with open("SampleData/capitals.json", encoding="utf8") as f:
    CAPITALS = json.loads(f.read())

# Initialse blank pipeline, add PhraseMatcher to match countries:
nlp = spacy.blank("en")
matcher = PhraseMatcher(nlp.vocab)
matcher.add(key="COUNTRY", docs=list(nlp.pipe(COUNTRIES)))


@Language.component("countries_component")
def countries_component_function(doc):
    """Use matcher to find all target spans, and label them with 'GPE'."""
    matches = matcher(doc)
    doc.ents = [Span(doc, start, end, label="GPE") for match_id, start, end in matches]
    return doc


# Add custom component to pipeline, uses matcher to label all countries with 'GPE'.
nlp.add_pipe(factory_name="countries_component")

# Get capital city from capitals dict, use as span extension getter:
get_capital = lambda span: CAPITALS.get(span.text)
Span.set_extension("capital", getter=get_capital)

print("\nnlp:")
print("->", nlp)

print("\nnlp.pipe_names:")
for ele in nlp.pipe_names:
    print("->", ele)

sample_text = "Czech Republic may help Slovakia protect its airspace"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

print("\nEntities (text, label, capital) in sample text:")
for ent in doc.ents:
    print("->", ent.text, ent.label_, ent._.capital)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.13")

# BAD:
# -> docs = [nlp(text) for text in LOTS_OF_TEXTS]

# GOOD:
# -> docs = list(nlp.pipe(LOTS_OF_TEXTS))
# -> The nlp.pipe method processes the texts as a stream and yields Doc objects.


Doc.set_extension("id", default=None)
Doc.set_extension("page_number", default=None)

data = [
    ("This is a text", {"id": 1, "page_number": 15}),
    ("And another text", {"id": 2, "page_number": 16}),
]

print(f"\nData inserted into nlp.pipe:\n->{data}")
print("\nData, Context in nlp.pipe:")
for doc, context in nlp.pipe(data, as_tuples=True):
    doc._.id = context["id"]
    doc._.page_number = context["page_number"]
    print(f"-> doc: {doc}, context: {context}")

# Can use just tokenizer, with:
# -> doc = nlp.make_doc("Hello world!").
# -> This turns text into Doc object.

# Can disable certain pipeline steps, temporarily:
# -> with nlp.select_pipes(disable=["tagger", "parser"]):
# ->     doc = nlp(text)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.14")

with open("SampleData/tweets.json", encoding="utf8") as f:
    TEXTS = json.loads(f.read())

nlp = spacy.load("en_core_web_sm")
print("\nAdjectives in tweets.json:")
for text in TEXTS:
    doc = nlp(text)
    print("->", [token.text for token in doc if token.pos_ == "ADJ"])

nlp = spacy.load("en_core_web_sm")
docs = [nlp(text) for text in TEXTS]
entities = [doc.ents for doc in docs]
print("\nEntities in tweets.json:")
for entity in entities:
    print("->", entity)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.15")

with open("SampleData/bookquotes.json", encoding="utf8") as f:
    DATA = json.loads(f.read())

nlp = spacy.blank("en")
Doc.set_extension("author", default=None)
Doc.set_extension("book", default=None)

for doc, context in nlp.pipe(DATA, as_tuples=True):
    doc._.book = context["book"]
    doc._.author = context["author"]
    print(f"\n{doc.text}\n — '{doc._.book}' by {doc._.author}")

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 3.16")

nlp = spacy.load("en_core_web_sm")
text = (
    "Chick-fil-A is an American fast food restaurant chain headquartered in "
    "the city of College Park, Georgia, specializing in chicken sandwiches."
    "Sentence number 2!"
)

doc = nlp(text)
print("\nToken-level features:")
for token in doc:
    print(f"-> Text: {token.text}, POS: {token.pos_}, Lemma: {token.lemma_}, "
          f"Is Stopword: {token.is_stop}, Dependency: {token.dep_}, "
          f"Shape: {token.shape_}, Is Alpha: {token.is_alpha}, "
          f"Is Punct: {token.is_punct}")

print("\nSentence-level features:")
for sent in doc.sents:
    print(f"-> Sentence: {sent.text}")

print("\nNamed Entities, Phrases, and Concepts:")
for ent in doc.ents:
    print(f"-> Text: {ent.text}, Label: {ent.label_}")

print("\nNoun Chunks:")
for chunk in doc.noun_chunks:
    print(f"-> Text: {chunk.text}, Root Text: {chunk.root.text}, "
          f"Root Dependency: {chunk.root.dep_}, "
          f"Root Head Text: {chunk.root.head.text}")

print("\nDependency Parse Tree:")
for token in doc:
    print(f"-> Text: {token.text}, Dependency: {token.dep_}, Head Text: {token.head.text}, "
          f"Children: {[child for child in token.children]}")
