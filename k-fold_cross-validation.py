from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

bc = load_breast_cancer()

datos = bc.data
salida = bc.target

# k-model
knn_model = KNeighborsClassifier()
resultados_kfold = cross_val_score(knn_model,datos,salida,cv=10,scoring='accuracy')
print(resultados_kfold.mean())

# k-model con diferente k vecinos
hiper_k_rango = range(1,30)
scores=[]
for k in hiper_k_rango:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    resultados_kfold = cross_val_score(knn_model,datos,salida,cv=10,scoring='accuracy')
    scores.append(resultados_kfold.mean())

print(scores)

# el objetivo es conseguir modelos capaces de generalizar lo mejor posible, y una buena señal de que esto 
# está ocurriendo es si la varianza observada de los resultados se estabiliza. Se observó una mejora entre
# la aproximación inicial y el 10-fold CV

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold,cross_val_score,RepeatedKFold,RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

bc = load_breast_cancer()

datos = bc.data
salida = bc.target

knn_model = KNeighborsClassifier()

RKFold = RepeatedKFold(n_splits=10,n_repeats=50)
resultados_RKfold = cross_val_score(knn_model,datos,salida,cv=RKFold,scoring='accuracy')

print(resultados_kfold.mean())
