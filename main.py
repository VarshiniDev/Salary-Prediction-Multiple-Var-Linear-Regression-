import pandas as pd
from sklearn import linear_model
import numpy as np
from word2number import w2n
import math

print("Original Data ------------------------------------------")
data = pd.read_csv("hiring.csv")
print(data)
print("Data filled with zero ------------------------------------------")
data.experience = data.experience.fillna("zero")
print(data)
print("Converting word to number ------------------------------------------")
data.experience = data.experience.apply(w2n.word_to_num)
print(data)
print("The mean value of test_score ------------------------------------------")
data_changed = math.floor(data.test_score.mean())
print(data_changed)
print("Filling with mean value ------------------------------------------")
data.test_score = data.test_score.fillna(data_changed)
print(data)
print("Multi varied Linear Regression ------------------------------------------")
reg = linear_model.LinearRegression()
reg.fit(data[['experience', 'test_score', 'interview_score']].values, data.salary)
ans = reg.predict([[2, 9, 6]])
print(ans)
