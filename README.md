# codealpha_talks
import pandas as p from sklearn import model_selection as m,preprocessing as s,ensemble as e,metrics as c d=p.read_csv("iris.csv") y=d.pop('Species') if'Id'in d:d.pop('Id') x,X,t,T=m.train_test_split(d,y) S=s.StandardScaler() P=e.RandomForestClassifier().fit(S.fit_transform(x),t).predict(S.transform(X)) print(c.classification_report(T,P))
