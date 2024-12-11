from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from src.model_validation import cross_validate_model


def train_and_optimize_model(X_train, X_test, y_train, y_test):
    """
    Train het model en voer hyperparameter tuning uit.
    Args:
        X_train (pd.DataFrame): Trainingsdata
        X_test (pd.DataFrame): Testdata
        y_train (pd.Series): Target variabelen voor training
        y_test (pd.Series): Target variabelen voor testen
    """

    rf = RandomForestRegressor(random_state=42)

    # Hyperparameters voor GridSearch
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    # Maak de gridsearch voor hyperparameter optimalisatie
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_absolute_error')

    # Start de hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Print de beste hyperparameters
    print("Beste hyperparameters gevonden:", grid_search.best_params_)

    # Voorspellen met het beste model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluatie van het model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")

    # Cross validatie
    cross_validate_model(best_model, X_train, y_train)
