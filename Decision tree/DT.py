from sklearn import datasets 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn.tree import DecisionTreeClassifier 
from plotting import plot_decision_region 
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz  


iris = datasets.load_iris()
x = iris.data[:,[2,3]]
y = iris.target  

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1,stratify=y)

tree = DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=1)
tree.fit(x_train,y_train)

x_combined = np.vstack((x_train,x_test))
y_combined = np.hstack((y_train,y_test))
plot_decision_region(x_combined,y_combined,classifier=tree,test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()


dot_data = export_graphviz(tree,filled=True,rounded = True,class_names=['Setosa','Versicolor','Verginica'],feature_names=['petal length ','petal_width'],out_file= None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')


