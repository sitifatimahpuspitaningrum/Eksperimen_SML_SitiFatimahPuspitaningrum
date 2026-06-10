import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data berhasil dimuat! Shape: {df.shape}")
    return df

def preprocess_data(df):
    #Hapus kolom ID
    df = df.drop(columns=['id'])
    
    #Hapus baris gender 'Other'
    df = df[df['gender'] != 'Other']
    
    #Isi missing values BMI dengan median
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    
    #Encoding kolom kategorikal
    le = LabelEncoder()
    cat_cols = ['gender', 'ever_married', 'work_type', 
                'Residence_type', 'smoking_status']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    #Pisahkan fitur dan target
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    
    #Scaling fitur numerik
    scaler = StandardScaler()
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    #Gabungkan kembali
    train_df = X_train.copy()
    train_df['stroke'] = y_train.values
    test_df = X_test.copy()
    test_df['stroke'] = y_test.values
    
    df_preprocessed = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Preprocessing selesai! Shape: {df_preprocessed.shape}")
    return df_preprocessed

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    input_path = "../stroke_raw/healthcare-dataset-stroke-data.csv"
    output_path = "../preprocessing/stroke_preprocessing.csv"
    
    df = load_data(input_path)
    df_preprocessed = preprocess_data(df)
    save_data(df_preprocessed, output_path)
    print("Automate preprocessing selesai!")