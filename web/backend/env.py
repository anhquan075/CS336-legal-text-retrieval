model_configs = {
    'PhoBERT': {
        "path_data_org": '/workingspace/data/zac2021-ltr-data/',
        "path_c_code": '/workingspace/data/zac2021-ltr-data/legal_corpus.json',
        "tokenizer": 'vi_tokenize',
        "topk": 300,
        "do_lower_case": True,
        "max_seq_length": 256,
        "path_preprocessed_data": '/workingspace/data/zalo-tfidfbm25150-full-pyvi/',
        "model_path": '/workingspace/settings/Tfbm150E5-full-phobert-pyvi42/models/checkpoint-21500',
    }
}
