#1.Import important liberaries
import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

#2. Create your connection.
con = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", con)

# Data processing 
print(df.shape)
print(df.columns)


print(df.info())
print("As we can see from info of dataframe that there null values in different column")
#Null values per columns
null_columns=df.columns[df.isnull().any()]
print(null_columns)

print('\nLets see object data type:-')

Category= df.select_dtypes(include=['object']).columns
print(Category)
print(df.attacking_work_rate.value_counts())
print(df.defensive_work_rate.value_counts())
