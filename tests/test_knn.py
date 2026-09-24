import numpy as np

from amlc.knn import knn_features


def unit_rows(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_train_rows_only_see_neighbours_from_other_folds():
    # rows 0 and 1 are identical twins but share fold 0, so row 0 must NOT see row 1's target
    emb = unit_rows([[1, 0], [1, 0], [0, 1], [0.1, 1]])
    y = np.array([10.0, 99.0, 1.0, 2.0])
    folds = np.array([0, 0, 1, 1])
    tr, _ = knn_features(emb, y, folds, emb[:1], k=1)
    assert tr["knn1_y"][0] != 99.0
    assert tr["knn1_y"][0] in (1.0, 2.0)


def test_test_rows_see_all_train_rows_and_find_the_twin():
    emb = unit_rows([[1, 0], [0, 1], [0.7, 0.7]])
    y = np.array([5.0, 50.0, 20.0])
    folds = np.array([0, 1, 2])
    _, te = knn_features(emb, y, folds, unit_rows([[0, 1]]), k=1)
    assert te["knn1_y"][0] == 50.0
    assert abs(te["knn1_sim"][0] - 1.0) < 1e-6


def test_mean_of_k_neighbours():
    emb = unit_rows([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]])
    y = np.array([1.0, 2.0, 3.0, 100.0])
    folds = np.array([0, 1, 2, 3])
    _, te = knn_features(emb, y, folds, unit_rows([[1, 0]]), k=3)
    assert te["knn3_mean"][0] == 2.0
