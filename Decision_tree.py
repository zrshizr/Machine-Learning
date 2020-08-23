# from sklearn  import tree 
# #创建训练集
# X= [[0,0],[1,1]]
# Y = [0,1]

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X,Y)
# #预测
# c = clf.predict([[3.,3.]])
# print(c)
# #预测每个类的概率
# d =  clf.predict_proba([[2., 2.]])
# print(d)

# 使用iris数据集
from sklearn.datasets import load_iris
from sklearn import tree 
X  , y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
tree.plot_tree(clf) 

import graphviz 
dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
dot_data = tree.export_graphviz(clf, out_file=None, 
                          feature_names=iris.feature_names,  
                          class_names=iris.target_names,  
                          filled=True, rounded=True,  
                          special_characters=True) 
graph = graphviz.Source(dot_data)  
graph                        

