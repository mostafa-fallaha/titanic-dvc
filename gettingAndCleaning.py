import io
import dvc.api, dvc.repo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path 
import subprocess

url = "https://github.com/mostafa-fallaha/titanic-dvc"
data = dvc.api.read("data/Titanic.csv", repo=url)
df = pd.read_csv(io.StringIO(data))

df.drop(columns=['Unnamed: 0', 'Name', 'SexCode'], axis=1, inplace=True)
df.fillna({'Age': df.Age.mean()}, inplace=True)
df = pd.get_dummies(data=df, columns=['PClass', 'Sex'], drop_first=True)

scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

columns_to_convert = ['PClass_2nd', 'PClass_3rd', 'Sex_male']
df[columns_to_convert] = df[columns_to_convert].astype(int)
print(df.head())

# Now the data is ready to be used by a Model

filepath = Path('data/Titanic.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath, index=False)


#------------------------- Versioning the data -----------------------------------

subprocess.run(["dvc", "add", "data/Titanic.csv"], check=True)
subprocess.run(["git", "add", "data/Titanic.csv.dvc", "data/.gitignore", "gettingAndCleaning.py"], check=True)
subprocess.run(["git", "commit", "-m", "cleaned data to be ready for a Model"], check=True)
subprocess.run(["dvc", "push"], check=True)
subprocess.run(["git", "push"], check=True)

