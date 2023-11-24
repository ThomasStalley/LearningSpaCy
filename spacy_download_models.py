import spacy


def download():
    """Function to be called via 'poetry run postinstall' to download desired spacy model(s)."""
    print("Downloading spaCy models...")
    spacy.cli.download("en_core_web_sm")
    print("-> en_core_web_sm succesfully downloaded.")
    spacy.cli.download("en_core_web_md")
    print("-> en_core_web_md succesfully downloaded.")
