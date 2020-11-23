from sklearn.tree import export_graphviz,DecisionTreeClassifier  
from pydotplus import graph_from_dot_data 
import collections 
from pydot import Dot

X = [ [180, 15,0],     
      [177, 42,0],
      [136, 35,1],
      [174, 65,0],
      [141, 28,1]]

Y = ['man', 'woman', 'woman', 'man', 'woman']    

data_feature_names = [ 'height', 'hair length', 'voice pitch' ]

clf = DecisionTreeClassifier().fit(X,Y)

dot_data = export_graphviz(clf,feature_names=data_feature_names,out_file=None,filled=True,rounded=True)
graph = graph_from_dot_data(dot_data)
#graph = Dot(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
        
graph.write_png('tree.bmp')