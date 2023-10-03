# Ex-06-Feature-Transformation
# AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
## STEP 1:
Read the given Data

## STEP 2:
Clean the Data Set using Data Cleaning Process

## STEP 3:
Apply Feature Transformation techniques to all the features of the data set

## STEP 4:
Print the transformed features

# PROGRAM:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```

# OUTPUT:

![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/3d174fb8-9a27-4f45-897e-7383dc655b22)


![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/2c3d8638-e097-4b97-a545-f9e497ace6f4)


![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/8b628854-f6dd-4a65-b6e3-f46c186d839f)


![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/65b6d287-8916-4ac6-b178-74f76860acc4)


![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/bd2b01d8-25e5-471e-affa-c634cd27efbf)


![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/af6fea14-5846-4287-87f0-7f08676bf745)


![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/7bc55760-90ba-4274-8659-02fcb49de1aa)


![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/08156362-bb56-45b7-bddf-273c3ff0ba6b)



![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/d3506781-05a4-41cf-b42f-41a74d26eeb1)


![image](https://github.com/vinushcv/Ex-06-Feature-Transformation/assets/113975318/0ac2bd4c-2e0c-472b-bd6a-5130e41600f0)


# RESULT:
Thus feature transformation is done for the given dataset.

