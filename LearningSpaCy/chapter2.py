import json

import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

# ------------------------------------------------------------------------------------------------------------------ 2.1
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.1")

# Vocab is used to store data used across multiple documents.
# Spacy encodes all strings to hash values, to use less memory.
# Strings are stored in StringStore using nlp.vocab.strings, in which we can also search for desired strings.

nlp = spacy.blank("en")
sample_text = "I love coffee"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

print("-> hash value:", nlp.vocab.strings["coffee"])
print("-> string value:", nlp.vocab.strings[3197928453018144401])

# A lexeme object, is an entry into the vocabulary.
# It contains the context-independent information about a word.

doc = nlp("I love coffee")
lexeme = nlp.vocab["coffee"]
print("-> lexeme.text:", lexeme.text)  # text
print("-> lexeme.orth:", lexeme.orth)  # hash
print("-> lexeme.is_alpha:", lexeme.is_alpha)  # example lexical attribute

# ------------------------------------------------------------------------------------------------------------------ 2.2
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.2")

nlp = spacy.blank("en")
sample_text = "I have a cat"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

cat_hash = nlp.vocab.strings["cat"]
cat_string = nlp.vocab.strings[cat_hash]
print("-> cat_hash:", cat_hash)
print("-> cat_string:", cat_string)

nlp = spacy.blank("en")
sample_text = "David Bowie is a PERSON"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

person_hash = nlp.vocab.strings["PERSON"]
person_string = nlp.vocab.strings[person_hash]
print("-> person_hash:", person_hash)
print("-> person_string:", person_string)

# ------------------------------------------------------------------------------------------------------------------ 2.4
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.4")

nlp = spacy.blank("en")

# Create a doc manually, using chosen words and spaces:
words = ["Hello", "world", "!"]
spaces = [True, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(f"\nManually created doc: \"{doc.text}\"")

span = Span(doc, 0, 2)
span_with_label = Span(doc, 0, 2, label="GREETING")
doc.ents = [span_with_label]

# Best practices (src: https://course.spacy.io/en/chapter2):
# -> Doc and Span are very powerful and hold references and relationships of words and sentences.
# -> Convert result to strings as late as possible.
# -> Use token attributes if available – for example, token.i for the token index.
# -> Don't forget to pass in the shared vocab.

# ------------------------------------------------------------------------------------------------------------------ 2.5
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.5")

nlp = spacy.blank("en")
words = ["spaCy", "is", "cool", "!"]
spaces = [True, True, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(f"\nManually created doc: \"{doc.text}\"")

nlp = spacy.blank("en")
words = ["Go", ",", "get", "started", "!"]
spaces = [False, True, True, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(f"\nManually created doc: \"{doc.text}\"")

nlp = spacy.blank("en")
words = ["Oh", ",", "really", "?", "!"]
spaces = [False, True, False, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(f"\nManually created doc: \"{doc.text}\"")

# ------------------------------------------------------------------------------------------------------------------ 2.6
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.6")

nlp = spacy.blank("en")

# Create a doc manually:
words = ["I", "like", "David", "Bowie"]
spaces = [True, True, True, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(f"\nManually created doc: \"{doc.text}\"")

# Create a span manually, and give it a label:
span = Span(doc, 2, 4, label="PERSON")
print(f"\nManually created span: \"{span.text}\"")

# Add the span to the doc's entities:
doc.ents = [span]
print("\nSpan added to doc's entities:", [(ent.text, ent.label_) for ent in doc.ents])

# ------------------------------------------------------------------------------------------------------------------ 2.7
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.7")

sample_text = "Berlin looks like a nice city"
print(f"\nSample text: \"{sample_text}\"")
nlp = spacy.load("en_core_web_sm")
doc = nlp(sample_text)

# Bad code:
token_texts = [token.text for token in doc]
part_of_speech_tags = [token.pos_ for token in doc]
for index, part_of_speech_tag in enumerate(part_of_speech_tags):
    if part_of_speech_tag == "PROPN":
        if part_of_speech_tags[index + 1] == "VERB":  # is next token a verb?
            result = token_texts[index]
            print("-> Found proper noun before a verb (using if part_of_speech_tag == \"PROPN\" logic):", result)

# Good code:
for token in doc:  # Loop through tokens in the doc
    if token.pos_ == "PROPN":  # is current token a proper noun?
        if token.i + 1 < len(doc):  # Ensure the next token index is within the document range!
            next_token = doc[token.i + 1]
            if next_token.pos_ == "VERB":  # is next token a verb?
                print("-> Found proper noun before a verb (using if token.pos_ == \"PROPN\" logic):", token.text)

# ------------------------------------------------------------------------------------------------------------------ 2.8
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.8")

# Word vectors and semantic similarity (src: https://course.spacy.io/en/chapter2):
# -> spaCy can compare two objects and predict similarity, using:
# ----> Doc.similarity(), Span.similarity() and Token.similarity()
# ----> Take another object and return a similarity score (0 to 1)
# -> Important: needs a pipeline that has word vectors included:
# ----> Can do this with en_core_web_md (medium)
# ----> Can do this with en_core_web_lg (large)
# ----> Can NOT do this with en_core_web_sm (small)

nlp = spacy.load("en_core_web_md")

# Compare two documents:
doc1 = nlp("I like fast food")
doc2 = nlp("I like pizza")
print(f"\nDoc 1: \"{doc1}\"")
print(f"Doc 2: \"{doc2}\"")
print("-> Similarity score:", doc1.similarity(doc2))

# Compare two tokens:
doc = nlp("I like pizza and pasta")
token1 = doc[2]
token2 = doc[4]
print(f"\nToken1: \"{token1}\"")
print(f"Token2: \"{token2}\"")
print("-> Similarity score:", token1.similarity(token2))

# Compare a document with a token:
doc = nlp("I like pizza")
token = nlp("soap")[0]
print(f"\nDoc: \"{doc}\"")
print(f"Token: \"{token}\"")
print("-> Similarity score:", doc.similarity(token))

# Compare a span with a document:
span = nlp("I like pizza and pasta")[2:5]
doc = nlp("McDonalds sells burgers")
print(f"\nSpan: \"{span}\"")
print(f"Doc: \"{doc}\"")
print("-> Similarity score:", span.similarity(doc))

# How does spaCy predict similarity?
# -> Similarity is determined using word vectors
# -> Multi-dimensional meaning representations of words
# -> Generated using an algorithm like Word2Vec and lots of text
# -> Can be added to spaCy's pipelines
# -> Default: cosine similarity, but can be adjusted
# -> Doc and Span vectors default to average of token vectors
# -> Short phrases are better than long documents with many irrelevant words

nlp = spacy.load("en_core_web_md")
sample_text = "I have a banana"
doc = nlp(sample_text)
chosen_word = doc[3]

print(f"\nSample text: \"{sample_text}\"")
print("-> chosen_word:", chosen_word.text)
print("-> chosen_word.vector:\n", chosen_word.vector)

# Similarity depends on the application context:
# -> Useful for many applications: recommendation systems, flagging duplicates etc.
# -> There's no objective definition of "similarity"
# -> Depends on the context and what application needs to do

nlp = spacy.load("en_core_web_md")
doc1 = nlp("I like cats")
doc2 = nlp("I hate cats")

print(f"\nDoc1: \"{doc1.text}\"")
print(f"Doc2: \"{doc2.text}\"")
print("-> Similarity score:", doc1.similarity(doc2))

# ------------------------------------------------------------------------------------------------------------------ 2.9
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.9")

nlp = spacy.load("en_core_web_md")
doc = nlp("Two bananas in pyjamas")
bananas_vector = doc[1].vector
print("\nVector the the token \"bananas\":\n", bananas_vector)

# ----------------------------------------------------------------------------------------------------------------- 2.10
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.10")

nlp = spacy.load("en_core_web_md")
doc1 = nlp("It's a warm summer day")
doc2 = nlp("It's sunny outside")
print(f"\nDoc1: \"{doc1}\"")
print(f"Doc2: \"{doc2}\"")
print("-> Similarity score:", doc1.similarity(doc2))

nlp = spacy.load("en_core_web_md")
doc = nlp("TV and books")
token1, token2 = doc[0], doc[2]
print(f"\nToken1: \"{token1}\"")
print(f"Token2: \"{token2}\"")
print("-> Similarity score:", token1.similarity(token2))

nlp = spacy.load("en_core_web_md")
doc = nlp("This was a great restaurant. Afterwards, we went to a really nice bar.")
span1 = doc[3:5]
span2 = doc[12:15]
print(f"\nSpan1: \"{span1}\"")
print(f"Span2: \"{span2}\"")
print("-> Similarity score:", span1.similarity(span2))

# ----------------------------------------------------------------------------------------------------------------- 2.11
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.11")

# Statistical models:
# -> Use cases: application needs to generalize based on examples
# -> Real-world examples: product names, person names, subject/object relationships
# -> spaCy features: entity recognizer, dependency parser, part-of-speech tagger

# Rule-based systems:
# -> Use cases: dictionary with finite number of examples
# -> Real-world examples: countries of the world, cities, drug names, dog breeds
# -> spaCy features: tokenizer, Matcher, PhraseMatcher

# Rule-based matching:
# -> Patterns are lists of dictionaries describing the tokens
# -> Operators (OP) can specify how often a token should be matched

love_cats_pattern = [{"LEMMA": "love", "POS": "VERB"}, {"LOWER": "cats"}]
very_happy_patterm = [{"TEXT": "very", "OP": "+"}, {"TEXT": "happy"}]

sample_text = "I love cats and I'm very very happy"
doc = nlp(sample_text)

matcher = Matcher(nlp.vocab)
matcher.add(key="LOVE_CATS", patterns=[love_cats_pattern])
matcher.add(key="VERY_HAPPY", patterns=[very_happy_patterm])
matches = matcher(doc)

sample_text = "I have a Golden Retriever"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

matcher = Matcher(nlp.vocab)
golden_retriever_pattern = [{"LOWER": "golden"}, {"LOWER": "retriever"}]
matcher.add(key="DOG", patterns=[golden_retriever_pattern])

for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print("-> Matched span (w/ Matcher):", span.text)
    print("-> Root token:", span.root.text)  # <- Get the span's root token and root head token
    print("-> Root head token:", span.root.head.text)
    print("-> Previous token:", doc[start - 1].text, doc[start - 1].pos_)  # <- Get the previous token and its POS tag

# PhraseMatcher:
# -> like regular expressions or keyword search – but with access to the tokens!
# -> Takes Doc object as patterns
# -> More efficient and faster than the Matcher
# -> Great for matching large word lists

sample_text = "I have a Golden Retriever"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

matcher = PhraseMatcher(nlp.vocab)
pattern = nlp("Golden Retriever")
matcher.add("DOG", [pattern])

for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print("-> Matched span (w/ PhraseMatcher):", span.text)

# ----------------------------------------------------------------------------------------------------------------- 2.12

# pattern = [{"LOWER": "silicon"}, {"LOWER": "valley"}]
# The pattern matches "SILICON VALLEY", or "Silicon Valley" etc, as when these are set to lower() they match the pattern
# Whitespaces dont need to be added to patterns, inbetween words/tokens

# ----------------------------------------------------------------------------------------------------------------- 2.13
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.13")

nlp = spacy.load("en_core_web_sm")
sample_text = (
    "Twitch Prime, the perks program for Amazon Prime members offering free "
    "loot, games and other benefits, is ditching one of its best features: "
    "ad-free viewing. According to an email sent out to Amazon Prime members "
    "today, ad-free viewing will no longer be included as a part of Twitch "
    "Prime for new members, beginning on September 14. However, members with "
    "existing annual subscriptions will be able to continue to enjoy ad-free "
    "viewing until their subscription comes up for renewal. Those with "
    "monthly subscriptions will have access to ad-free viewing until October 15."
)
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

amazon_prime_pattern = [{"LOWER": "amazon"}, {"IS_TITLE": True, "POS": "PROPN"}]
add_free_viewing_pattern = [{"LOWER": "ad"}, {"IS_PUNCT": True}, {"LOWER": "free"}, {"POS": "NOUN"}]
matcher = Matcher(nlp.vocab)
matcher.add(key="AMAZON_PRIME_PATTERN", patterns=[amazon_prime_pattern])
matcher.add(key="AD_FREE_VIEWING_PATTERN", patterns=[add_free_viewing_pattern])

for match_id, start, end in matcher(doc):
    print(f"-> Pattern Key: \"{doc.vocab.strings[match_id]}\", Matched Span: \"{doc[start:end].text}\"")

# ----------------------------------------------------------------------------------------------------------------- 2.14
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.14")

with open("SampleData/countries.json", encoding="utf8") as file:
    COUNTRIES = json.loads(file.read())

nlp = spacy.blank("en")
sample_text = "Czech Republic may help Slovakia protect its airspace"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

matcher = PhraseMatcher(nlp.vocab)
patterns = list(nlp.pipe(COUNTRIES))
matcher.add("COUNTRY", patterns)
matches = matcher(doc)
for match_id, start, end in matches:
    print(f"-> Matched country from sample_text: {doc[start:end]}")

# ----------------------------------------------------------------------------------------------------------------- 2.15
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 2.15")

with open("SampleData/prose.txt", encoding="utf8") as file:
    TEXT = file.read()

nlp = spacy.load("en_core_web_sm")

sample_text = TEXT
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(TEXT)

matcher = PhraseMatcher(nlp.vocab)
patterns = list(nlp.pipe(COUNTRIES))
matcher.add("COUNTRY", patterns)

# Create span from matches, label each with "GPE" (geopolitical entity):
doc.ents = []
for match_id, start, end in matcher(doc):
    span = Span(doc, start, end, label="GPE")
    print(f"-> span.root.head.text: {span.root.head.text}, span.text: {span.text}")
    doc.ents = list(doc.ents) + [span]
