import pandas as pd
import numpy as np

#data = [1,2,3,4,5]
#data = [['alex',10],['sifat',25],['masuk',20]]
'''data = [{'a':1,'b':2},{'c':3,'d':4}]
df = pd.DataFrame(data)
print(df)'''
dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }
df = pd.DataFrame(dict)
df.index = ['BR','RU','IN','CH','SA']
print(df)