# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

## REG NO:212224220019
## NAME: DEEPAK JG
```
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv('/content/bmi (1) (1).csv')
df
```
![Screenshot 2025-04-20 061215](https://github.com/user-attachments/assets/74c15923-3851-4ae8-b017-81161980c090)
```
df.head()
```

![Screenshot 2025-04-20 061222](https://github.com/user-attachments/assets/f7c2bb0a-14ba-45bf-b057-00f0f0b6b55b)
```
df.dropna()
```

![Screenshot 2025-04-20 061228](https://github.com/user-attachments/assets/eefc9cad-c1db-486e-a7d9-113a9bbf306e)
```
max_vals=np.max(np.abs(df[['Height','Weight']]),axis=0)
max_vals
```

![Screenshot 2025-04-20 061235](https://github.com/user-attachments/assets/ef4a6add-73aa-415f-83e7-7ba58deb3525)
```
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![Screenshot 2025-04-20 061240](https://github.com/user-attachments/assets/94deb295-932a-4d5f-ad94-039bd7ac70e8)
```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

![Screenshot 2025-04-20 061246](https://github.com/user-attachments/assets/4f9b7538-6dc9-4794-b37f-fc64e46e55fe)
```
from sklearn.preprocessing import Normalizer
norm=Normalizer()
df[['Height','Weight']]=norm.fit_transform(df[['Height','Weight']])
df
```

![Screenshot 2025-04-20 061428](https://github.com/user-attachments/assets/7332fd56-2cd5-4dd6-abd9-3d34417f9ffb)
```
df3=pd.read_csv('/content/bmi (1) (1).csv')
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
````

![Screenshot 2025-04-20 061433](https://github.com/user-attachments/assets/083e666b-80bb-4b14-8c73-dbdd642f737f)
```
df4=pd.read_csv('/content/bmi (1) (1).csv')
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4
```

![Screenshot 2025-04-20 061439](https://github.com/user-attachments/assets/51a86c0c-8442-4743-961f-b67462158c7f)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![Screenshot 2025-04-20 061443](https://github.com/user-attachments/assets/4d8dcb28-5f00-45f9-87d7-74f1f167088f)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![Screenshot 2025-04-20 061448](https://github.com/user-attachments/assets/1805fccb-2fbd-4d87-b35c-fa44a0241dd5)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print('Chi-squared statistic:',chi2)
print('p-value:',p)
```

![Screenshot 2025-04-20 061453](https://github.com/user-attachments/assets/ce73efc2-f7fe-477a-937b-2fe3e7381c09)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
     'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print('Selected Features:')
print(selected_features)
```

![Screenshot 2025-04-20 061457](https://github.com/user-attachments/assets/1e66e6b5-3ed5-4b64-8135-61b54c419ba1)



# RESULT:
       # INCLUDE YOUR RESULT HERE
