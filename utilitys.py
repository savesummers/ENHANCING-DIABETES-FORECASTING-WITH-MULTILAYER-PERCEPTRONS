import pandas as pd
import numpy as np
from sklearn import preprocessing

class Utilitys:

	def preprocess(self, df):
		print('Before preprocessing')
		print('Number of rows with zero as a variable')
		for col in df.columns:
			missing_rows = df.loc[df[col] == 0].shape[0]
			print(col + ': ' + str(missing_rows))

		df['Glucose'] = df['Glucose'].replace(0, np.nan)
		df['Pregnancies'] = df['Pregnancies'].replace(0, np.nan)
		df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
		df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
		df['Insulin'] = df['Insulin'].replace(0, np.nan)
		df['BMI'] = df['BMI'].replace(0, np.nan)
		df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
		df['Pregnancies'] = df['Pregnancies'].fillna(df['Pregnancies'].mean())
		df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
		df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
		df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
		df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

		print('After preprocessing')
		print('Number of rows with zero as a variable')
		for col in df.columns:
			missing_rows = df.loc[df[col] == 0].shape[0]
			print(col + ': ' + str(missing_rows))

		df_scaled = preprocessing.scale(df)
		df_scaled = pd.DataFrame(df_scaled, columns = df.columns)
		df_scaled['Outcome'] = df['Outcome']
		df = df_scaled

		print(df.describe().loc[['mean','std','max'],].round(2).abs())
		print(df.describe())
		
		return df_scaled
		
	
df = pd.read_csv('diabetes.csv')
utility = Utilitys()
df_processed = utility.preprocess(df)
