#Initialise python packages
#pip install pandas numpy scikit-learn matplotlib seaborn

#importing csvs
import pandas as pd
import glob

# Path to folder containing all CSV files
path = "C:/apt/CICIDS2017/*.csv"

files = glob.glob(path)

df_list = []

for file in files:
    print(f"Loading {file}")
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

# Combine all into one dataframe
df = pd.concat(df_list, ignore_index=True)

print("Final shape:", df.shape)

#
print(df.shape)
print(df.head())
print(df['Label'].value_counts())
