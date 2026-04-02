import pandas as pd


def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    return df


def clean_target_variable(df):
    """
    Convert loan_status into binary target
    """
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()

    df['target'] = df['loan_status'].map({
        'Fully Paid': 0,
        'Charged Off': 1
    })

    return df


def basic_cleaning(df):

    # Convert interest rate
    if df['int_rate'].dtype == 'object':
        df['int_rate'] = df['int_rate'].str.replace('%', '', regex=False).astype(float)

    # Convert revol_util
    if df['revol_util'].dtype == 'object':
        df['revol_util'] = df['revol_util'].str.replace('%', '', regex=False).astype(float)

    # Convert term (e.g. "36 months" -> 36)
    df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(int)

    return df