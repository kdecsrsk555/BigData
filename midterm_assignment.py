import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(csv_file_path):
    """
    실세계 데이터 CSV 파일을 읽어 전처리 파이프라인을 수행하는 함수.
    
    Parameters:
        csv_file_path (str): 입력 CSV 파일 경로.
        
    Returns:
        pd.DataFrame: 전처리된 데이터프레임.
        dict: 전처리 과정에서 생성된 객체(예: 스케일러, 인코더).
    """
    
    # 1. CSV 파일 읽기
    try:
        df = pd.read_csv(csv_file_path)
        print("CSV 파일 읽기 완료. 데이터프레임 shape:", df.shape)
    except Exception as e:
        raise Exception(f"CSV 파일 읽기 오류: {e}")
    
    # 2. 중복 데이터 제거
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    print(f"중복 제거 후 데이터프레임 shape: {df.shape} (제거된 행: {initial_rows - df.shape[0]})")
    
    # 3. 오류 데이터 제거 (예: 'age' 열이 음수인 경우)
    # 주의: 실제 데이터에 따라 조건을 수정해야 함
    if 'age' in df.columns:
        df = df[df['age'] >= 0]
        print(f"오류 데이터('age' < 0) 제거 후 shape: {df.shape}")
    
    # 4. 결측치 처리
    # 수치형 열: 평균으로 대체
    # 범주형 열: 최빈값으로 대체
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("결측치 처리 완료. 남은 결측치:", df.isnull().sum().sum())
    
    # 5. 이상치 탐지 및 처리 (IQR 방법)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 이상치를 중앙값으로 대체
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        df.loc[outliers, col] = df[col].median()
        print(f"{col} 열에서 이상치 {outliers.sum()}개 처리")
    
    # 6. 범주형 인코딩 (원-핫 인코딩)
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        
        # 원-핫 인코딩된 데이터프레임 생성
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        print("범주형 변수 인코딩 완료. 새로운 열:", encoded_cols)
    else:
        encoder = None
        print("범주형 변수 없음. 인코딩 스킵.")
    
    # 7. 스케일링 (StandardScaler)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("수치형 변수 스케일링 완료.")
    
    # 8. 파생 변수 생성
    # 예: 두 수치형 열의 비율 생성 (실제 데이터에 따라 수정 필요)
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['feature1_to_feature2_ratio'] = df['feature1'] / (df['feature2'] + 1e-6)  # 0으로 나누기 방지
        print("파생 변수 'feature1_to_feature2_ratio' 생성")
    
    # 전처리 객체 저장
    preprocessing_objects = {
        'scaler': scaler,
        'encoder': encoder
    }
    
    return df, preprocessing_objects

# 사용 예시
if __name__ == "__main__":
    # CSV 파일 경로 지정 (실제 경로로 수정 필요)
    csv_path = "sample_data.csv"
    
    try:
        processed_df, preprocess_objects = preprocess_data(csv_path)
        print("\n전처리된 데이터프레임 미리보기:")
        print(processed_df.head())
        print("\n전처리 객체:", preprocess_objects)
    except Exception as e:
        print(f"오류 발생: {e}")