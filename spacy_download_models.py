import spacy


def download_spacy_models():
    """Download specified spaCy models - called via 'poetry run postinstall'."""
    models = ["en_core_web_sm", "en_core_web_md"]
    print("Downloading spaCy models...")
    for model in models:
        try:
            spacy.cli.download(model)
            print(f"-> {model} successfully downloaded.")
        except Exception as e:
            print(f"Unable to download {model}. Error: {e}")
