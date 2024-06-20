"""
Fit a model to predict the price of an apartment in Paris.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import joblib
import logging


# Data source
FILE = (
    "https://files.data.gouv.fr/geo-dvf/latest/csv/2022/"
    "departements/75.csv.gz"
)


def prepare_data(df):
    """
    Preprocess the data.
    """
    df = df[df.nature_mutation == "Vente"]  # Keep only sales
    df = df[df.type_local == "Appartement"]  # Keep only apartments
    df = df[df.nature_culture.isna()]  # Keep only urban properties
    df.code_postal = df.code_postal.astype(int)  # Convert zip code to int

    # Create new variable: total surface of the lots
    total_surface_lots = (
        df.lot1_surface_carrez.fillna(0) +
        df.lot2_surface_carrez.fillna(0) +
        df.lot3_surface_carrez.fillna(0) +
        df.lot4_surface_carrez.fillna(0) +
        df.lot5_surface_carrez.fillna(0)
    )
    df["total_surface_lots"] = total_surface_lots

    # Select relevant columns
    df = df[[
        "valeur_fonciere",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "code_postal",
        "total_surface_lots"
    ]]

    df = df.dropna()  # Drop missing values

    # Remove outliers
    quartiles = df.valeur_fonciere.quantile([0.05, 0.25, 0.5, 0.75])
    iqd = quartiles[0.75] - quartiles[0.25]
    df = df[
        (df.valeur_fonciere < quartiles[0.75] + 0.5 * iqd) &
        (df.valeur_fonciere > quartiles[0.05])
    ]

    # Print some statistics of the transformed data
    logging.debug("Data transformation completed")
    logging.debug("Shape of the data")
    logging.debug(f"{df.shape}")

    logging.debug("Min and max of the target variable")
    logging.debug(f"{df.valeur_fonciere.min()}, {df.valeur_fonciere.max()}")

    return df


def main():

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] %(asctime)s %(message)s"
        )

    df = pd.read_csv(FILE, compression="gzip", low_memory=False)
    df = prepare_data(df)

    transformer = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="error"), ["code_postal"])
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessing", transformer),
        ("model", RandomForestRegressor())
    ])

    # Prepare training and test
    X_df = df.drop(columns="valeur_fonciere")
    y_df = df.valeur_fonciere

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
        )

    pipeline.fit(X_train, y_train)

    # Evaluate the model
    yhat = pipeline.predict(X_test)

    mape = ((yhat - y_test).abs() / y_test).mean()
    r2 = pipeline.score(X_test, y_test)
    aes = (yhat - y_test).abs()

    logging.info(f"Median absolute error: {aes.median():.2f}")
    logging.info(f"MAPE: {mape:.2f}")
    logging.info(f"R2: {r2:.2f}")

    logging.debug(transformer.named_transformers_["ohe"].categories_)
    # Save the model to a file
    joblib.dump(pipeline, "pipeline.joblib")


if __name__ == "__main__":
    main()
