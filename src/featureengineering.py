def select_features(df):

    features = [
        'loan_amnt',
        'term',
        'int_rate',
        'emp_length',
        'home_ownership',
        'annual_inc',
        'verification_status',
        'purpose',
        'dti',
        'delinq_2yrs',
        'revol_util',
        'open_acc'
    ]

    X = df[features]
    y = df['target']

    return X, y