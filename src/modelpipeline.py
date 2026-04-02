from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def create_pipeline():

    numeric_features = [
        'loan_amnt',
        'int_rate',
        'annual_inc',
        'dti',
        'delinq_2yrs',
        'revol_util',
        'open_acc',
        'term'
    ]

    categorical_features = [
        'emp_length',
        'home_ownership',
        'verification_status',
        'purpose'
    ]

    numeric_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(max_iter=1000))
        ]
    )

    return pipeline