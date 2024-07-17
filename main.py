import pandas as pd
from sklearn import linear_model
import numpy as np
# from word2number import w2n
import math

data = pd.read_csv("hiring.csv")
print(data)
print("------------------------------------------")
data.experience = data.experience.fillna("zero")
print(data)
print("------------------------------------------")
data.experience = data.experience.apply(w2n.word_to_num)
print(data)
# d.experience = d.experience.apply(w2n.word_to_num)
print("------------------------------------------")
data_changed = math.floor(data.test_score.mean())
print(data_changed)
data.test_score = data.test_score.fillna(data_changed)
print(data)

reg = linear_model.LinearRegression()
reg.fit(data[['experience', 'test_score', 'interview_score']].values, data.salary)
ans = reg.predict([[2, 9, 6]])
print(ans)
