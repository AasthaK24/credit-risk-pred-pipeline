import sys
sys.path.append("src")

from datapreprocessing import load_data, clean_target_variable, basic_cleaning
from featureengineering import select_features

import pandas as pd


def main():

    print("Loading data...")

    df = load_data("data/loan.csv")

    print("Cleaning target variable...")

    df = clean_target_variable(df)

    print("Running basic cleaning...")

    df = basic_cleaning(df)

    print("Selecting features...")

    X, y = select_features(df)

    cleaned_df = pd.concat([X, y], axis=1)

    print("Saving cleaned dataset...")

    cleaned_df.to_csv("data/cleaned_loan.csv", index=False)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()