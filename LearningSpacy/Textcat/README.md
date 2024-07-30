_**text extraction example project**_

process:
1. data prepared with `python prepare_data.py`
2. base config created with spacy quickstart tool
3. config created with `python -m spacy init fill-config ./base_config.cfg ./config.cfg`
4. model trained with `python -m spacy train config.cfg --output ./output`
5. model tested with `python test_model.py`

src: https://www.youtube.com/watch?v=7PD48PFL9VQ