import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('C:\Users\WZU448\ct16_cap1_ds4\project_2\data\heart_disease')

col_names=['age','sex','cp','trestbps','chol','fbs','restecg',
'thalach','exang','oldpeak','slope','ca','thal','num']

df_cle = pd.read_csv('processed.cleveland.data', header=None, names=col_names, na_values='?')
df_hung = pd.read_csv('processed.hungarian.data', header=None, names=col_names, na_values='?')
df_switz = pd.read_csv('processed.switzerland.data', header=None, names=col_names, na_values='?')
df_va = pd.read_csv('processed.va.data', header=None, names=col_names, na_values='?')

df = pd.concat([df_cle, df_hung, df_switz, df_va])

#Dummy - cp, restecg, slope, ca, thal
#Interactions - oldpeak*slope

#Explore missings
df.isnull().sum()

df.count(axis=1).value_counts()

highly_missing = ['slope','ca','thal'] #41,44,51

df.drop(highly_missing, axis=1).count(axis=1).value_counts()


#Thal - best strategy seems to be to fill with 3

##impute target

df.loc[df['num']>0, 'num'] = 1

##Pairplot

df_drop = df.drop(highly_missing, axis=1).dropna()
plt.figure()
#sns.pairplot(data=df_drop, x_vars=['num'], y_vars=df_drop.drop('num').columns,
#            dropna=True)
#plt.figure()
#sns.pairplot(data=df_drop, x_vars=['age','trestbps','chol','thalach','oldpeak'],
#             y_vars=df_drop.columns, hue='num')
             
#variables of interest: age, chol, thalach, cp, trestbps
             
df.groupby(['num','slope']).size()
#slope seems important, might impute 0 for missings

df.groupby(['num','ca']).size()
#also seems important

df.groupby(['num','thal']).size()
#also important

######################################################
#Decision Tree

df[['slope','ca','thal']] = df[['slope','ca','thal']].fillna(99)

df.groupby(['num','slope']).size()
df.groupby(['num','ca']).size()
df.groupby(['num','thal']).size()

##numeric trestbps, chol, thalach, oldpeak

num_columns = ['trestbps','chol','thalach','oldpeak']
medians =df[num_columns].median()

df[num_columns] = df[num_columns].fillna(medians)

df.isnull().sum()

df[['fbs','restecg','exang']] = df[['fbs','restecg','exang']].fillna(99)

#Make Tree

def dtree_gen(data, Y_col, depth):
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    Y = data[Y_col]
    X = data.drop(Y_col, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3, random_state = 42)
    dtree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced")
    dtree.fit(X_train,Y_train)
    return dtree
    

def dtree_acc(data, Y_col, depth):
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    Y = data[Y_col]
    X = data.drop(Y_col, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3, random_state = 11)
    dtree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced")
    dtree.fit(X_train,Y_train)
    Y_pred = dtree.predict(X_test)
    return accuracy_score(Y_test, Y_pred)

for i in range(1,21):
    print i, dtree_acc(df, 'num', i)
    
def dtree_cross(data, Y_col, depth):
    from sklearn.cross_validation import cross_val_predict
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    Y = data[Y_col]
    X = data.drop(Y_col, axis=1)
    dtree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced")
    Y_pred = cross_val_predict(dtree,X,Y,cv=3)
    return accuracy_score(Y, Y_pred)

    
for i in range(1,21):
    print i, 'cross', dtree_cross(df, 'num', i), 'standard', dtree_acc(df, 'num', i)
    
#5 is the best with a single train/test split
    
dtree = dtree_gen(df, 'num', 5)

def get_code(tree, feature_names):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node):
                if (threshold[node] != -2):
                        print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        print "} else {"
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        print "}"
                else:
                        print "return " + str(value[node])

        recurse(left, right, threshold, features, 0)

get_code(dtree, df.columns)
    
#from sklearn.externals.six import StringIO
#with open("iris.dot", 'w') as f:
#    f = tree.export_graphviz(dtree, out_file=f)
#    
#from sklearn.externals.six import StringIO  
#import pydot 
#dot_data = StringIO() 
#tree.export_graphviz(clf, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("iris.pdf") 
    
#cp, chol, thal, ca, oldpeak, slope

df_dropnas = df.drop(['slope','ca','thal'], axis=1)
    
dtree = dtree_gen(df_dropnas, 'num', 3)
get_code(dtree, df_dropnas.columns)

for i in range(1,21):
    print i, 'cross', dtree_cross(df_dropnas, 'num', i), 'standard', dtree_acc(df_dropnas, 'num', i)
    
#cp, chol, exang
    
########After using the decision tree, going to use logistic regression
    
df = pd.concat([df_cle, df_hung, df_switz, df_va])

#only keep hypothesized valuable columns, drop exang NAs
columns_keep = ['age','cp','chol','exang','oldpeak','num','thalach']
df = df[columns_keep]
df.loc[df['chol']==0, 'chol'] = np.NaN
df = df.dropna()

df.isnull().sum()

df.loc[df['num']>0, 'num'] = 1

#df.groupby('num').size()
#df.groupby(['cp','num']).size()

#sns.pairplot(data=df, x_vars=['age','chol'], y_vars = ['age','chol'], hue='num')

##Dummy for CP
cp_dummy = pd.get_dummies(df['cp'], prefix = 'cp')
df = pd.concat([df, cp_dummy[['cp_3.0','cp_4.0']]], axis=1)
df=df.drop('cp',axis=1)

##Dummy for Age
df['age60'] = 1
df.loc[df['age']<60, 'age60'] = 0

##Dummy for exang
df = pd.concat([df, pd.get_dummies(df['exang'], prefix = 'exang')], axis=1)
df = df.drop(['exang','exang_0.0'],axis=1)

#Center and scale age and chol
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(df[['age','chol','oldpeak','thalach']])
df[['age','chol','oldpeak','thalach']] = scaler.transform(df[['age','chol','oldpeak','thalach']])

df['cp4']=df['cp_4.0']
df['exang1']=df['exang_1.0']
df = df.drop(['cp_4.0','exang_1.0'],axis=1)

#Interaction
import patsy as pat

df = pd.concat([df, pat.dmatrix("age:chol + age:thalach + age:cp4 + age:exang1 - 1", df, return_type='dataframe')], axis=1,)

##Statsmodels
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

x_drop = ['num','cp_3.0', 'age60','chol','age:chol', 'age:thalach', 'age:cp4', 'age:exang1']

Y = df.num
X = sm.add_constant(df.drop(x_drop,axis=1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3, random_state = 11)
logreg = sm.Logit(Y_train,X_train).fit()
logreg.summary()
predicted = logreg.predict(X_test)
predicted_train = logreg.predict(X_train)
Y_predlog = (predicted > .2).astype(int)
Y_predlog_train = (predicted_train > .2).astype(int)
accuracy_score(Y_test, Y_predlog)
print precision_recall_fscore_support(Y_test, Y_predlog, average='binary')
print precision_recall_fscore_support(Y_train, Y_predlog_train, average='binary')


dtree = DecisionTreeClassifier(max_depth=3, class_weight="balanced")
dtree.fit(X_train,Y_train)
Y_predtree = dtree.predict(X_test)
accuracy_score(Y_test, Y_predtree)
print precision_recall_fscore_support(Y_test, Y_predtree, average='binary')

forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, Y_train)
Y_predforest = dtree.predict(X_test)
accuracy_score(Y_test, Y_predforest)
print precision_recall_fscore_support(Y_test, Y_predforest, average='binary')

nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_prednb = nb.predict(X_test)
accuracy_score(Y_test, Y_prednb)
print precision_recall_fscore_support(Y_test, Y_prednb, average='binary')

###Now we're going to make an ROC curve and cross validate the model quality scores

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

###############################################################################
# Data IO and generation

# import some data to play with
X = df.drop(x_drop,axis=1).values
y = df.num.values
n_samples, n_features = X.shape

###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(y, n_folds=10)
classifier = LogisticRegression()

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []


for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC for Hospital Data')
plt.legend(loc="lower right")
plt.savefig('roc_curve_nonstrat.png')
plt.show()

###########################3
#See ROC Thresholds

fpr_list = []
tpr_list = []
thresholds_list = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    thresholds_list.append(thresholds)
    
print fpr_list[0][0], tpr_list[0][0], thresholds_list[0][0]

 #0.0 0.0185185185185 0.989919611802

thresholds_df = pd.DataFrame({'fpr':fpr_list[0], 'tpr':tpr_list[0], 'thresholds':thresholds_list[0]})

#############################
#CV Accuracy, Recall, Fscore

def prec_rec_fscore(threshold):
    prec = []
    recall = []
    fscore = [] 
    for i, (train, test) in enumerate(cv):
        classifier.fit(X[train], y[train])
        predictions = np.array([x[1] for x in classifier.predict_proba(X[test])])
        Y_pred = (predictions > threshold).astype(int)
        values = precision_recall_fscore_support(y[test], Y_pred, average='binary')
        prec.append(values[0])
        recall.append(values[1])
        fscore.append(values[2])
    prec_mean = np.array(prec).mean()
    recall_mean = np.array(recall).mean()
    fscore_mean = np.array(fscore).mean()
    return prec_mean, recall_mean, fscore_mean

prec_rec_fscore(.5)
    
thresholds = np.arange(.05,.55,.05)

for number in thresholds:
    print "threshold = " + str(number) + "  precision, recall, fscore = " + str(prec_rec_fscore(number))
    
for number in thresholds:
    precv, recv, fscorev = prec_rec_fscore(number)
    prec.append(precv)
    recall.append(recv)
    fscore.append(fscorev)
    
plt.plot(thresholds, fscore)
plt.plot(thresholds, recall)
    

prec = []
recall = []
fscore = [] 
    
for i, (train, test) in enumerate(cv):
    classifier.fit(X[train], y[train])
    predictions = np.array([x[0] for x in classifier.predict_proba(X[test])])
    Y_pred = (predictions > .2).astype(int)
    values = precision_recall_fscore_support(y[test], Y_pred, average='binary')
    prec.append(values[0])
    recall.append(values[1])
    fscore.append(values[2])
    
for i, (train, test) in enumerate(cv):
    print i, train, test


