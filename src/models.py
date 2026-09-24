"""
models.py

Training routines for the two AI models used in the fraud detection pipeline:
Logistic Regression (interpretable baseline) and Random Forest (stronger ensemble).
"""

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm


def train_logistic_regression(
    X_train, y_train, random_state: int = 42, max_iter: int = 1000, verbose: bool = True
) -> LogisticRegression:
    """Train a Logistic Regression fraud classifier, with basic progress logging."""
    if verbose:
        print("Training Logistic Regression...")
    t0 = time.time()

    model = LogisticRegression(
        max_iter=max_iter, random_state=random_state, verbose=1 if verbose else 0
    )
    model.fit(X_train, y_train)

    if verbose:
        print(f"Logistic Regression trained in {time.time() - t0:.2f} seconds.")
    return model


def train_random_forest(
    X_train,
    y_train,
    total_trees: int = 200,
    step: int = 20,
    max_depth: int = 12,
    random_state: int = 42,
    show_progress: bool = True,
) -> RandomForestClassifier:
    """
    Train a Random Forest fraud classifier, building trees incrementally
    (warm_start) so training progress can be shown via a tqdm progress bar.
    """
    if show_progress:
        print("Training Random Forest...")
    t0 = time.time()

    model = RandomForestClassifier(
        n_estimators=step,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        warm_start=True,
    )

    tree_counts = range(step, total_trees + 1, step)
    iterator = tqdm(tree_counts, desc="Random Forest trees") if show_progress else tree_counts

    for n_trees in iterator:
        model.n_estimators = n_trees
        model.fit(X_train, y_train)

    if show_progress:
        print(f"Random Forest trained in {time.time() - t0:.2f} seconds ({model.n_estimators} trees).")
    return model
