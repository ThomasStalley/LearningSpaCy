import spacy
from spacy.matcher import Matcher

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.1")
# Create a blank English nlp object, input sample text:
nlp = spacy.blank("en")

# Create doc by inserting text into model:
sample_text = "Hello world!"
print(f"\nSample text: \"{sample_text}\"\n")
doc = nlp(sample_text)

# Doc is made of tokens separated by whitespace:
for token in doc:
    print("token:", token.text)

# A slice of the Doc is a span:
span = doc[1:3]
print("\nspan:", span.text)

# We can reveal lexical attributes of the tokens:
doc = nlp("It costs $5.")
print("\nIndex:   ", [token.i for token in doc])
print("Text:    ", [token.text for token in doc])
print("is_alpha:", [token.is_alpha for token in doc])
print("is_punct:", [token.is_punct for token in doc])
print("like_num:", [token.like_num for token in doc])

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.2")
# Create the German nlp object:
nlp = spacy.blank("de")

sample_text = "Liebe Grüße!"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)
print("\nDE doc:", doc.text)

# Create the Spanish nlp object:
nlp = spacy.blank("es")

sample_text = "¿Cómo estás?"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)
print("\nES doc:", doc.text)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.3")

nlp = spacy.blank("en")

sample_text = "I like tree kangaroos and narwhals."
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

tree_kangaroos = doc[2:4]
print("\ntree_kangaroos span:", tree_kangaroos.text)

tree_kangaroos_and_narwhals = doc[2:]
print("\ntree_kangaroos_and_narwhals span:", tree_kangaroos_and_narwhals.text, "\n")

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.4")
nlp = spacy.blank("en")

sample_text = "In 1990, more than 60% of people in East Asia were in extreme poverty. Now less than 4% are."
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

# Iterate over the tokens in the doc, to find percentages:
for token in doc:
    if token.like_num:
        next_token = doc[token.i + 1]
        if next_token.text == "%":
            print("Percentage found:", token.text)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.5")
# Load in English pipeline:
nlp = spacy.load("en_core_web_sm")

sample_text = "She ate the pizza"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

# Iterate over the tokens
for token in doc:
    print("\nToken text:", token.text)
    print("Predicted part-of-speech tag:", token.pos_)
    print("Predicted syntactic dependencies:", token.dep_)
    print("Dependent/Attached word:", token.head.text)

# Process sample text:, then iterate through tokens to give predicted entity labels:
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    print("\nEntity:", ent.text)
    print("Predicted label:", ent.label_)
    print("Explained label:", spacy.explain(ent.label_))

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.8")
nlp = spacy.load("en_core_web_sm")
sample_text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

token, pos_tag, dep_label = "TOKEN:", "POS TAG:", "DEPENDENCY LABEL:"
print(f"\n{token:<12}{pos_tag:<10}{dep_label:<10}")
for token in doc:
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    print(f"{token_text:<12}{token_pos:<10}{token_dep:<10}")

# Iterate over the entities:
for ent in doc.ents:
    print("\nEntity:", ent.text)
    print("Predicted label:", ent.label_)
    print("Explained label:", spacy.explain(ent.label_))

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.9")

nlp = spacy.load("en_core_web_sm")
sample_text = "Upcoming iPhone X release date leaked as Apple reveals pre-orders"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

# Iterate over the entities:
for ent in doc.ents:
    print("\nEntity:", ent.text)
    print("Predicted label:", ent.label_)

# Get the span for "iPhone X":
iphone_x = doc[1:3]
print("\nMissing entity:", iphone_x.text)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.10")

# Matching with chosen words:
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

patterns = [[{"TEXT": "iPhone"}, {"TEXT": "X"}]]
matcher.add(key="IphonePattern", patterns=patterns)

sample_text = "Upcoming iPhone X release date leaked"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("--> Matched text using matcher:", matched_span.text)

# Matching with lexical attribute:
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

patterns = [[{"IS_DIGIT": True}, {"LOWER": "fifa"}, {"LOWER": "world"}, {"LOWER": "cup"}, {"IS_PUNCT": True}]]
matcher.add(key="WorldCupPattern", patterns=patterns)

sample_text = "2018 FIFA World Cup: France won!"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("--> Matched text using matcher:", matched_span.text)

# Matching with token attributes:
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

patterns = [[{"LEMMA": "love", "POS": "VERB"}, {"POS": "NOUN"}]]
matcher.add(key="AnimalLovePattern", patterns=patterns)

sample_text = "I loved dogs but now I love cats more."
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("--> Matched text using matcher:", matched_span.text)

# Matching with operators and quantifiers (where "OP": "?" below indicates match of 0 or 1 times):
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

patterns = pattern = [[{"LEMMA": "buy"}, {"POS": "DET", "OP": "?"}, {"POS": "NOUN"}]]
matcher.add(key="QuantifierPattern", patterns=patterns)

sample_text = "I bought a smartphone. Now I'm buying apps."
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("--> Matched text using matcher:", matched_span.text)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.11")

nlp = spacy.load("en_core_web_sm")
sample_text = "Upcoming iPhone X release date leaked as Apple reveals pre-orders"
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

matcher = Matcher(nlp.vocab)
iphone_pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add(key="IphoneXPattern", patterns=[iphone_pattern])

matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("--> Matched text using matcher:", matched_span.text)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - CHAPTER 1.12")

# Matching IOS Version:
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

sample_text = ("After making the iOS update you won't notice a radical system-wide "
               "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
               "iOS 11's furniture remains the same as in iOS 10. But you will discover "
               "some tweaks once you delve a little deeper.")
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

ios_version_pattern = [{"TEXT": "iOS"}, {"IS_DIGIT": True}]
matcher.add(key="IosVersionPattern", patterns=[ios_version_pattern])
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("--> Matched text using matcher:", matched_span.text)

# Matching Downloaded Software:
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

sample_text = ("i downloaded Fortnite on my laptop and can't open the game at all. Help? "
               "so when I was downloading Minecraft, I got the Windows version where it "
               "is the '.zip' folder and I used the default program to unpack it... do "
               "I also need to download Winzip?")
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

downloaded_software_pattern = [{"LEMMA": "download"}, {"POS": "PROPN"}]
matcher.add(key="DownloadedSoftwarePattern", patterns=[downloaded_software_pattern])
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("--> Matched text using matcher:", matched_span.text)

# Matching adjective then noun (then optional noun):
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

sample_text = ("Features of the app include a beautiful design, smart search, automatic "
               "labels and optional voice responses.")
print(f"\nSample text: \"{sample_text}\"")
doc = nlp(sample_text)

adjective_then_noun_pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]
matcher.add(key="AdjectiveThenNounPattern", patterns=[adjective_then_noun_pattern])
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("--> Matched text using matcher:", matched_span.text)
