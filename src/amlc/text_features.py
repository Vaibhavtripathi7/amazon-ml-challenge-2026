"""Low-memory TF-IDF text matrix: word 1-2 grams + hashed char 3-4 grams from the first 600 chars."""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer


def tfidf_matrices(text_tr, text_te):
    """Fit on train text only (IDF uses no target, so no leakage). Returns (X_train, X_test) CSR float32."""
    text_tr, text_te = text_tr.fillna(""), text_te.fillna("")
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True, dtype=np.float32)
    char_hash = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=2**19,
                                  alternate_sign=False, norm=None, dtype=np.float32)
    char_idf = TfidfTransformer(sublinear_tf=True)
    head = lambda s: s.str.slice(0, 600)  # char pieces from the start only (item name + first details): caps memory
    X_tr = hstack([word.fit_transform(text_tr), char_idf.fit_transform(char_hash.transform(head(text_tr)))]).tocsr()
    X_te = hstack([word.transform(text_te), char_idf.transform(char_hash.transform(head(text_te)))]).tocsr()
    return X_tr, X_te
