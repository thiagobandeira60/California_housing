# %% markdown
# # California Housing Prices Project
# %%
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
# %%
df = pd.read_csv('housing.csv')
# %%
df.head()
# %%
df.info()
# %%
df.isnull().sum()
# %% markdown
# ### The median_house_value column is a continuous variable, so we are talking about a regression problem. We are going to follow the below steps:
#
# #### 1. Initial EDA to have a glimpse of the data distribution and the data itself
# #### 2. We will deal with the missing data
# #### 3. Model test and evaluation
# #### 4. Model Selection and improval
# #### 5. Conclusion
# %% markdown
# ### 1. Initial EDA
# %%
# Let's use a pairplot to have a general idea of the distribution of the data.
sns.pairplot(df, height=3.5)
plt.tight_layout()
# %%
# Here we are creating a correlation matrix to understand the relationship among variables.
df.corr().style.background_gradient()
# %%
# We can also use a heatmap to see that

sns.set(context="paper",font="monospace")
df_corr_matrix = df.corr()
fig, axe = plt.subplots(figsize=(15,8))
cmap = sns.diverging_palette(220,10,center = "light", as_cmap=True)
sns.heatmap(df_corr_matrix,vmax=1,square =True, cmap=cmap,annot=True);
# %%
df['total_rooms'].describe()
# %%
df['total_bedrooms'].describe()
# %%
df['median_house_value'].describe()
# %%
sns.boxplot(data=df)
plt.gcf().set_size_inches(15,8)
# %%
# As we saw, there are a ouliers in the column total_rooms.
# Outliers lie i a distance of 1.5 times the inter quantile range (1.5*IQR)

def getOutliers(dataframe,column):
    column = "total_rooms"
    #df[column].plot.box(figsize=(8,8))
    des = dataframe[column].describe()
    desPairs = {"count":0,"mean":1,"std":2,"min":3,"25":4,"50":5,"75":6,"max":7}
    Q1 = des[desPairs['25']]
    Q3 = des[desPairs['75']]
    IQR = Q3-Q1
    lowerBound = Q1-1.5*IQR
    upperBound = Q3+1.5*IQR
    print("(IQR = {})\nOutliers are anything outside this range: ({},{})".format(IQR,lowerBound,upperBound))
    #b = df[(df['a'] > 1) & (df['a'] < 5)]
    data = dataframe[(dataframe [column] < lowerBound) | (dataframe [column] > upperBound)]

    print("The total number of outliers from our dataset of {} rows are:\n{}".format(df[column].size,len(data[column])))
    #remove the outliers from the dataframe
    outlierRemoved = df[~df[column].isin(data[column])]
    return outlierRemoved
# %%
# Now let's remove the outliers

df_outliersRemoved = getOutliers(df,"total_rooms")
# %% markdown
# ### 2. Dealing with missing Data
# %%
# Let's see how data is distributed

bedrooms = df.loc[(df['total_bedrooms'].notnull()), 'total_bedrooms']
# %%
bedrooms.hist(figsize=(12,8),bins=50)
# %%
print(df.iloc[:,4:5].head())
imputer = SimpleImputer(np.nan,strategy ="median")
imputer.fit(df.iloc[:,4:5])
df.iloc[:,4:5] = imputer.transform(df.iloc[:,4:5])
df.isnull().sum()
# %%
# Let's take a look at the variable ocean_proximity, which is categorical

df.ocean_proximity.unique()
# %%
# Let's encode it

labelEncoder = LabelEncoder()
df["ocean_proximity"].value_counts()
df["ocean_proximity"] = labelEncoder.fit_transform(df["ocean_proximity"])
df["ocean_proximity"].value_counts()
df.describe()
# %%
df.ocean_proximity.unique()
# %%
df.head()
# %% markdown
# ### 3. Model test and evaluation
# %%
# Here we are going to try Linear regression, Decision tree regression, and random forest regression.
# After we see which one performs best, we can play with the variables and with scaling to see if we can improve the model.
# %% markdown
# #### 3.1. Linear Regression
# %%
X = df.drop("median_house_value",axis=1)
y = df["median_house_value"]

print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# %%
# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# %%
# Our R-squared was of 0.6137. This number is basically saying that our current model (using all the independent
# variables as predictors) can explain about 61.37% of our data.
# %%
# Here we have some plots on how the predicted model compares to the actual one.

test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',);
# %% markdown
# #### 3.2. Decision Tree Regression
# %%
dtReg = DecisionTreeRegressor(max_depth=9)
dtReg.fit(X_train,y_train)
# %%
dtReg_y_pred = dtReg.predict(X_test)
dtReg_y_pred
# %%
# Compute and print R^2 and RMSE
print("R^2: {}".format(dtReg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, dtReg_y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# %%
test = pd.DataFrame({'Predicted':dtReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
# %% markdown
# #### 3.3. Random Forest Regression
# %%
rfReg = RandomForestRegressor(30)
rfReg.fit(X_train,y_train)
# %%
rfReg_y_pred = rfReg.predict(X_test)
# %%
# Compute and print R^2 and RMSE
print("R^2: {}".format(rfReg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, rfReg_y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# %%
test = pd.DataFrame({'Predicted':rfReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
# %% markdown
# ### 4. Model Selection and Improval
# %% markdown
# #### Considering the tested models, we can conclude that the Random Forest Regression Model can explain more the data than the other models with an R-squared of 0.805. We will now take this model and change some parameters such as variables, scaling, etc.to see if we can improve the model to explain more of the data.
# %%
df.corr().style.background_gradient()
# %%
# Considering the correlation matrix, the variables that correlate more with median_house_value are:
# housing_median_age, total_rooms, and median_income
# Let's see how the model performs with those variables
# %%
data = df[['median_house_value', 'housing_median_age', 'total_rooms', 'median_income']]
# %%
data.head()
# %%
X = data.drop("median_house_value",axis=1)
y = data["median_house_value"]

print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# %%
rfReg = RandomForestRegressor(30)
rfReg.fit(X_train,y_train)
# %%
rfReg_y_pred = rfReg.predict(X_test)
# %%
# Compute and print R^2 and RMSE
print("R^2: {}".format(rfReg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, rfReg_y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# %%
test = pd.DataFrame({'Predicted':rfReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
# %%
# By limiting the model to 3 variables we actually had a worse performance than before.
# %%
# 2 different approaches we can try is to log normalize the total_rooms column and to standardize it, applying the changes
# to the whole dataset to see if the model improves.
# %%
df['total_rooms'].var()
# %%
df['total_rooms_normalized'] = np.log(df['total_rooms'])
# %%
df['total_rooms_normalized'].var()
# %%
df.head()
# %%
df2 = df.drop('total_rooms', axis=1)
# %%
df2.head()
# %%
# Now let's see the result of normalizing the column total_rooms
# %%
X = df2.drop("median_house_value",axis=1)
y = df2["median_house_value"]

print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# %%
rfReg = RandomForestRegressor(30)
rfReg.fit(X_train,y_train)
# %%
rfReg_y_pred = rfReg.predict(X_test)
# %%
# Compute and print R^2 and RMSE
print("R^2: {}".format(rfReg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, rfReg_y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# %%
test = pd.DataFrame({'Predicted':rfReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
# %%
# We slightly improved the model by having a higher R-squared of 0.809 and a smaller root mean squared error.
# %%
# Now let's try to standardize the dataset by using the StandardScaler
# %%
df3 = df.drop('total_rooms_normalized', axis=1)
# %%
df3.head()
# %%
ss = StandardScaler()
# %%
df3_scaled = ss.fit_transform(df3[['housing_median_age', 'total_rooms', 'total_bedrooms', 'median_income']])
# %%
X_train, X_test, y_train, y_test = train_test_split(df3_scaled, y, test_size=0.2, random_state=42)
# %%
rfReg = RandomForestRegressor(30)
rfReg.fit(X_train,y_train)
# %%
rfReg_y_pred = rfReg.predict(X_test)
# %%
# Compute and print R^2 and RMSE
print("R^2: {}".format(rfReg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, rfReg_y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# %%
test = pd.DataFrame({'Predicted':rfReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
# %%
# By subseting the dataset and scaling it, we got a worse result.
# %%
# Another thing we can try is to see if we have other columns with a high variance and normalize them.
# %%
df.describe()
# %%
df['population'].var()
# %%
df['total_bedrooms'].var()
# %%
df4 = df.drop('total_rooms', axis=1)
# %%
df4.head()
# %%
df4['population_normalized'] = np.log(df4['population'])
# %%
df4['population_normalized'].var()
# %%
df4['total_bedrooms_normalized'] = np.log(df4['total_bedrooms'])
# %%
df4['total_bedrooms_normalized'].var()
# %%
df4.drop(columns=['total_bedrooms', 'population'], axis=1, inplace=True)
# %%
df4.head()
# %%
X = df4.drop("median_house_value",axis=1)
y = df4["median_house_value"]

print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# %%
rfReg = RandomForestRegressor(30)
rfReg.fit(X_train,y_train)
# %%
rfReg_y_pred = rfReg.predict(X_test)
# %%
# Compute and print R^2 and RMSE
print("R^2: {}".format(rfReg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, rfReg_y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# %%
test = pd.DataFrame({'Predicted':rfReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
# %% markdown
# ### 5. Conclusion
# %%
# What we conclude is that the best model is the Random Forest Regression.
# The model improves a little more when we normalize the total_rooms column, which has the highest variance amongst all
# columns.
# the model has:

# R-squared - 0.8098
# Root mean squared error - 49919.067

# Which means that our model can predict about 81% of the data we have
# %%
