import numpy as no
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR

# load the data
df = pd.read_csv('airfoil_self_noise.dat', sep = '\t', header = None)

# data consists of 5 columns representing input and then a final column that represents the output
# get the input
# iloc allows you to select sub-matrices of the dataframe,
# in this case we pick all rows but only up to the second to last column
data = df.iloc[:, :-1].values

# get ouput
# once again we use iloc, this time we only pick the last column.
target = df.iloc[:, -1:].values

# we split the data into train and test sets
X_train, X_test, y_train, y_test = tts(data, target, test_size=0.25)

# initialize model and train
model = LR()
model.fit(X_train, y_train)

# score model
model.score(X_test, y_test)


