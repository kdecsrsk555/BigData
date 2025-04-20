from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

# 샘플 데이터 구성
data = {
    'Age': [25, np.nan, 35, 45, np.nan],
    'Salary': [50000, 60000, np.nan, 80000, 90000],
    'Experience': [1, 3, 5, 7, 9]
}

df = pd.DataFrame(data)

# KNNImputer 객체 생성 (이웃 수 = 2)
imputer = KNNImputer(n_neighbors=2)

# 결측치 대체 수행 --> .fit_transform
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(df_imputed)