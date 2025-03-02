from sklearn.model_selection import cross_val_score


def cross_validate_model(model, X, y, cv=5):
    """
    Voer cross-validatie uit op het model.
    Args:
        model: Het model dat geverifieerd moet worden
        X (pd.DataFrame): Feature set
        y (pd.Series): Target labels
        cv (int): Aantal folds voor cross-validatie
    Returns:
        cv_scores (array): De cross-validatie scores
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")

    return cv_scores
