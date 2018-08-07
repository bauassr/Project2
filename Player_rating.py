#1.Import important liberaries
import sqlite3
import pandas as pd
 
# Core Libraries - Machine Learning
import sklearn

## Importing train_test_split,cross_val_score,GridSearchCV,KFold, RandomizedSearchCV - Validation and OptimizationC
from sklearn.model_selection import ShuffleSplit, train_test_split,cross_val_score,GridSearchCV,KFold, RandomizedSearchCV

# Importing Regressors - Modelling
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

# Importing Regression Metrics - Performance Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pickle
from math import sqrt
import matplotlib.pyplot as plt

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
df.dropna(axis=0,inplace=True)

print('\n let see the distribution of every column')
df.hist(bins=50,figsize=(20,15))
plt.show()
print('\nLets see object data type:-')
Category= df.select_dtypes(include=['object']).columns
print(Category)
print(df.attacking_work_rate.value_counts())
print("I guess these columns could play vitual role in prediction so lets convert these categories")
# for clear insite Plotting the distribution of the values in the attacking_work_rate column
df["attacking_work_rate"].value_counts().plot.bar()
# Choosing to replace only with low because it can improve the variance of the column
df.replace( ['None','norm','y','stoc','le'],'low', inplace = True)
print(df["attacking_work_rate"].value_counts())
df["attacking_work_rate"].value_counts().plot.bar()

print(df.defensive_work_rate.value_counts())
# Plotting the distribution of the values in the defensive_work_rate column
df["defensive_work_rate"].value_counts().plot.bar()
df.replace(['o', '1', '2', 'ormal', '3', '0', 'es', 'tocky', 'ean'],'low',inplace = True) 
df.replace(['5',  '6', '4'],'medium', inplace = True) 
df.replace([ '7', '9', '8'],'high', inplace = True) 
print(df["defensive_work_rate"].value_counts())
df["defensive_work_rate"].value_counts().plot.bar()
import seaborn as sns
# Checking for correlations using HEATMAP
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), cmap="PRGn")
df.corr().loc['overall_rating']
print("\noverall_rating is highly correlated with the reactions and potential columns(Correlation>0.7). It is moderately correlated with short_passing, long_passing,ball_control, shot_power,vision (correlation >0.4")

print("\nLet Prepare X and Y for Model")
#Prepare X first 
X = df.drop("overall_rating",axis = 1)
X.shape, X.columns

#Convert Data Column for better understanding 
X['year'] = pd.DatetimeIndex(X.date).year
X['month'] = pd.DatetimeIndex(X.date).month
X['day'] = pd.DatetimeIndex(X.date).day
X.drop('date',axis=1, inplace=True)


# Now Converting all Columns to Numerical for the shake of anaylisis
X_cat_cols = X.select_dtypes(include='object').columns.tolist()
X_cat_cols
# LabelEncoding the preferred_foot, attacking_work_rate, defensive_work_rate
from sklearn.preprocessing import LabelEncoder
for i in X_cat_cols:
    lbl_enc = LabelEncoder()
    X[i] = lbl_enc.fit_transform(X[i])
    
#Final Look to X variable 
# Checking the columns and the shape of the input vector after encoding
X.columns, X.shape    

# its turn for Y 
Y = df["overall_rating"]
Y.shape

print("\nSpliting test and train sets")
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.75, random_state = 100)
import math
print("\n Creating a model and checking model Score \n")
lm  = LinearRegression()
model  = lm.fit(x_train,y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
    
print('Linear Regression -', 'RMSE Train:', math.sqrt(mean_squared_error(y_train_pred, y_train)))
print('Linear Regression -', 'RMSE Test:' ,math.sqrt(mean_squared_error(y_test_pred, y_test)))  
print('Linear Regression -', 'R2_score Train:', r2_score(y_train_pred, y_train))
print('Linear Regression -', 'R2_score Test:' ,r2_score(y_test_pred, y_test))
#All Regressor together refrence Praveen
regressors = [
            ("Linear - ", LinearRegression(normalize=True)),
            ("Ridge - ",  Ridge(alpha=0.5, normalize=True)),
            ("Lasso - ",  Lasso(alpha=0.5, normalize=True)),
            ("ElasticNet - ",  ElasticNet(alpha=0.5, l1_ratio=0.5, normalize=True)),
            ("Decision Tree - ",  DecisionTreeRegressor(max_depth=5)),
            ("Random Forest - ",  RandomForestRegressor(n_estimators=100)),
            ("AdaBoost - ",  AdaBoostRegressor(n_estimators=100)),
            ("GBM - ", GradientBoostingRegressor(n_estimators=100))]
for reg in regressors:
    reg[1].fit(x_train, y_train)
    y_test_pred= reg[1].predict(x_test)
    print(reg[0],"\n\t R2-Score:", reg[1].score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")
print("\nfeature Selection using random forest \n")
rndf = RandomForestRegressor(n_estimators=150)
rndf.fit(x_train, y_train)
importance = pd.DataFrame.from_dict({'cols':x_train.columns, 'importance': rndf.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)
plt.figure(figsize=(20,15))
sns.barplot(importance.cols, importance.importance)
plt.xticks(rotation=90)

imp_cols = importance[importance.importance >= 0.005].cols.values
imp_cols

print("\nFitting models with columns where feature importance>=0.005\n")
x_train, x_test, y_train, y_test = train_test_split(X[imp_cols],Y,test_size=0.75, random_state = 100)
for reg in regressors:
    reg[1].fit(x_train, y_train)
    y_test_pred= reg[1].predict(x_test)
    print(reg[0],"\n\t R2-Score:", reg[1].score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")
    
imp_cols = importance[importance.importance >= 0.001].cols.values
imp_cols
print("\nFitting models with columns where feature importance>=0.001\n")
x_train, x_test, y_train, y_test = train_test_split(X[imp_cols],Y,test_size=0.75, random_state = 100)
for reg in regressors:
    reg[1].fit(x_train, y_train)
    y_test_pred= reg[1].predict(x_test)
    print(reg[0],"\n\t R2-Score:", reg[1].score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")
    

print("\nRandomForest and GBM provide us with the best RMSE and R2-Score when selecting columns with feature importance >= 0.001\n")

print("\nValidating our models using K-Fold Cross Validation for Robustness\n")
scoring = 'neg_mean_squared_error'
results=[]
names=[]
for modelname, model in regressors:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_train,y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(modelname)
    print(modelname,"\n\t CV-Mean:", cv_results.mean(),
                    "\n\t CV-Std. Dev:",  cv_results.std(),"\n")
    
print("\nRandomForest and GBM provide us with the best validation score, both w.r.t. CV-Mean and CV-Std. Dev\n")

print("""\n\nTherefore we choose these two models to optimize. 
      We do this by finding best hyper-parameter values which give us even better R2-Score and RMSE values""")    

print("\nTuning Model for better Performance -- Hyper-Parameter Optimization\n")

RF_Regressor =  RandomForestRegressor(n_estimators=100, n_jobs = -1, random_state = 100)

CV = ShuffleSplit(test_size=0.25, random_state=100)

param_grid = {"max_depth": [5, None],
              "n_estimators": [50, 100, 150, 200],
              "min_samples_split": [2, 4, 5],
              "min_samples_leaf": [2, 4, 6]
             }

rscv_grid = GridSearchCV(RF_Regressor, param_grid=param_grid, verbose=1)
rscv_grid.fit(x_train, y_train)
rscv_grid.best_params_
model = rscv_grid.best_estimator_
model.fit(x_train, y_train)
model.score(x_test, y_test)
RF_reg = pickle.dumps(rscv_grid)
#Gradient
GB_Regressor =  GradientBoostingRegressor(n_estimators=100)

CV = ShuffleSplit(test_size=0.25, random_state=100)

param_grid = {'max_depth': [5, 7, 9],
              'learning_rate': [0.1, 0.3, 0.5]
             }
rscv_grid = GridSearchCV(GB_Regressor, param_grid=param_grid, verbose=1)
rscv_grid.fit(x_train, y_train)
rscv_grid.best_params_
model = rscv_grid.best_estimator_
model.fit(x_train, y_train)
GB_reg = pickle.dumps(rscv_grid)

print("\nComparing performance metric of the different models")
RF_regressor = pickle.loads(RF_reg)
GB_regressor = pickle.loads(GB_reg)

print("RandomForestRegressor - \n\t R2-Score:", RF_regressor.score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(RF_regressor.predict(x_test), y_test)),"\n")
      
print("GradientBoostingRegressor - \n\t R2-Score:", GB_regressor.score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(GB_regressor.predict(x_test), y_test)),"\n")

print("""Choosing the model
We can see that Gradient Boosting Regressor gives better result with an R2-Score of more than 97% and while keeping RMSE value low(=1.1370474). So, XGBoost Regressor should be used as the regression model for this dataset""")
