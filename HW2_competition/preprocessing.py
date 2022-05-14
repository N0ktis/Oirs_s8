import pickle as pk
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import datetime as dt
from model import predict


def get_tfidf(data, vocab_path):
    tfidf_vocab = pk.load(open(vocab_path, 'rb'))
    tfidf_vectorizer = TfidfVectorizer(vocabulary=tfidf_vocab.vocabulary_)
    df_ = pd.DataFrame(tfidf_vectorizer.fit_transform(data).todense())
    return df_


def get_PCA(df, pca_path):
    pca = pk.load(open(pca_path, 'rb'))
    df_ = pd.DataFrame(pca.transform(df))
    return df_


def create_dataset(df):
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def get_prediction(file_path, model_path, vocab_path, pca_path, threshold):
    try:
        model = torch.load(model_path)
        check_df = pd.read_csv(file_path["requests"])
        check_df_vec = get_tfidf(check_df, vocab_path)
        check_df_pca = get_PCA(check_df_vec, pca_path)
        check_dataset, _, _ = create_dataset(check_df_pca)
        prediction, pred_loss = predict(model, check_dataset)

        result = pd.DataFrame({'y_true': [0 if l <= threshold else 1 for l in pred_loss]})

        file_name = 'Prediction_{time}.csv'.format(time=dt.datetime.now().strftime("%Y%m%d_%I:%M:%S"))
        result.to_csv(file_name)
        return file_name
    except Exception:
        print("Something went wrong.\n")
