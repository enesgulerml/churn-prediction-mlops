import pandas as pd
import numpy as np
from src.preprocessing import prepare_data


def test_prepare_data_logic():
    """
    Test if 'prepare_data' correctly removes the 'customerid' column
    and maps the target variable 'churn' to binary integers.
    """
    # 1. Create Dummy Data
    data = {
        'customerid': ['123', '456'],
        'gender': ['Male', 'Female'],
        'churn': ['Yes', 'No'],
        'tenure_months': [1, 10]
    }
    df = pd.DataFrame(data)

    # 2. Execute the function
    X, y = prepare_data(df)

    # 3. Assertions

    # Assert 'customerid' is dropped (Critical for avoiding data leakage)
    assert 'customerid' not in X.columns, "customerid should be removed from features"

    # Assert Target Mapping (Yes -> 1, No -> 0)
    assert y.iloc[0] == 1, "Churn 'Yes' should be mapped to 1"
    assert y.iloc[1] == 0, "Churn 'No' should be mapped to 0"

    # Assert Dimensions
    assert len(X) == 2
    assert len(y) == 2