import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def preprocess_data(file_path):
    """
    Laad en preprocessed de data set.
    Args:
        file_path (str): Pad naar het CSV-bestand.
    Returns:
        X_train (pd.DataFrame): Trainingsdata
        X_test (pd.DataFrame): Testdata
        y_train (pd.Series): Target variabelen voor training
        y_test (pd.Series): Target variabelen voor testen
    """
    df_shopping = pd.read_csv(file_path)

    # Zijn er missing values
    if df_shopping.isna().any().any():
        print("Er zijn ontbrekende waarden in het dataframe")
    elif df_shopping.isnull().any().any():
        print("Er zijn nul waardes in je dataset")
    else:
        print("Er zijn geen ontbrekende & 0 waarden in het dataframe")

    # Purchase amount als target variabele zetten
    X = df_shopping.drop(columns=['Purchase Amount (USD)'])
    y = df_shopping['Purchase Amount (USD)']

    # Train, split maken
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Categorische variabelen identificeren
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Numerieke kolommen
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Voor numerieke data: miswaarden imputeren en standaardiseren
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Voor categorische data: OneHotEncoder gebruiken
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # De complete preprocessing pipeline voor alle kolommen
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Zorgt dat alle data numerical wordt in de train en test set
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Verkrijg de feature-namen van de categorische kolommen na OneHotEncoding
    categorical_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)

    # Combineer de numerieke kolomnamen met de gecodeerde categorische feature-namen
    all_feature_names = numerical_cols + list(categorical_feature_names)

    # Zet de numpy arrays om in DataFrame met de juiste kolomnamen
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    # Maak een DataFrame met de juiste kolomnamen
    X_train = pd.DataFrame(X_train_dense, columns=all_feature_names)
    X_test = pd.DataFrame(X_test_dense, columns=all_feature_names)

    return X_train, X_test, y_train, y_test
