import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
# %matplotlib inline

import tensorflow.keras as tf

coverages_df = pd.read_csv("coverages.csv") 
employees_df = pd.read_csv("employees.csv")
employees_df.head()

# employees_df = employees_df.map(lambda x: {
#     "employee_id": x["ID"],
#     "coverage_id": x["Coverage ID"],
#     "age": x["Age"],
#     "gender": x["Gender"]
# })

print(employees_df.shape)

Xtrain, Xtest = train_test_split(employees_df, test_size=0.2, random_state=1)
print(f"Shape of train data: {Xtrain.shape}")
print(f"Shape of test data: {Xtest.shape}")

employee_id = employees_df.ID.nunique()
coverage_id = employees_df["Coverage ID"].nunique()