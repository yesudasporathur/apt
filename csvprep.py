import pandas as pd
import numpy as np
import glob



# LOAD MULTIPLE CSV FILES
import os

def csvLoader(folder_path):
    # Search recursively for ALL csv files
    files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    
    print(f"Found {len(files)} CSV files")
    
    if len(files) == 0:
        raise ValueError(" No CSV files found. Check your folder path.")
    
    df_list = []
    
    for file in files:
        try:
            print(f"Loading: {file}")
            
            temp_df = pd.read_csv(file, low_memory=False)
            temp_df.columns = temp_df.columns.str.strip()
            
            df_list.append(temp_df)
        
        except Exception as e:
            print(f"Skipping {file}: {e}")
    
    if len(df_list) == 0:
        raise ValueError("CSV files found but none could be loaded.")
    
    df = pd.concat(df_list, ignore_index=True)
    
    print("Merged successfully:", df.shape)
    
    return df

df = csvLoader("C:/apt/datasets/CICIDS2017")
#df = csvLoader("C:/apt/datasets/UNSW_NB15")


def cleanData(df):
    print("\nCleaning data...\n")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop missing values
    df.dropna(inplace=True)
    
    print("After cleaning:", df.shape)
    
    return df

df = cleanData(df)

def processLabels(df):
    print("\nProcessing labels...")
    
    # Convert to binary: BENIGN = 0, Attack = 1
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    print(df['Label'].value_counts())
    
    return df

df = processLabels(df)


df.to_csv("input.csv", index=False)
