import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Load Raw Data

df=pd.read_excel("RAW.xlsx")
df=df.dropna(how='any')
df['Visit_Realization']=df.Actual_Visit/df.Visit_Target
df['Sales_Potential'] = (1-df.Sales_X_Subcity_Q1)/df.Sales_Total_Subcity_Q1
df['Customer_Sales_Potential']=df.Customer_Potential/df.Market_Size_Subcity
df['Customer_Sales']=df.Customer_Sales_Potential*df.Sales_X_Subcity_Q1
df.head()



#Conver from Capital to Numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['Brand Awareness'])
le.classes_
le.transform(df['Brand Awareness'])
df['Brand Awareness'] = le.transform(df['Brand Awareness'])

data= df[['Visit_Realization','Brand Awareness','Customer_Potential']].values
subdf=df[['Visit_Realization','Sales_Potential','Brand Awareness','Market_Size_Subcity','Customer_Potential','Sales_Total_Subcity_Q1','Sales_X_Subcity_Q1','Customer_Sales_Potential','Customer_Sales']]
Decision_Tree_Var = df[['Customer_Sales_Potential','Market_Size_Subcity']]

#Creation Fuction of Calculation and Visualize Corellation

def visualize_corellation(data):
    import numpy as np
    correlations = data.corr()
    names =data.columns
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, data.columns.shape[0], 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()

#Corellation Overall Data
visualize_corellation(subdf)


#Selection the Model for Predition the Sales

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net",
         "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    GaussianNB(),
   ]

X,y = Decision_Tree_Var.values , df['Customer_Sales'].values
y= y.astype('int')
linearly_separable = (X, y)


datasets = [
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()



#Applied Decision Tree Clasification

#Visualize Classifier

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, clim=(y.min(), y.max()), zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X,y = Decision_Tree_Var.values , df['Customer_Sales'].values
y= y.astype('int')
tree = DecisionTreeClassifier().fit(X, y)
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1)
bag.fit(X, y)
y_p=bag.predict(X)


visualize_classifier(bag, X, y)
visualize_classifier(DecisionTreeClassifier(), X, y)
visualize_classifier(MLPClassifier(alpha=1, max_iter=1000), X, y)

#Load Test Data
test=pd.read_excel("RAWtest.xlsx")
test=test.dropna(how='any')
test['Visit_Realization']=test.Actual_Visit/df.Visit_Target
test['Sales_Potential'] = (1-df.Sales_X_Subcity_Q1)/df.Sales_Total_Subcity_Q1
test['Customer_Sales_Potential']=test.Customer_Potential/df.Market_Size_Subcity
test['Customer_Sales']=test.Customer_Sales_Potential*df.Sales_X_Subcity_Q1
test.head()


#Conver from Capital to Numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test['Brand Awareness'])
le.classes_
le.transform(test['Brand Awareness'])


datatest= test[['Visit_Realization','Brand Awareness','Customer_Potential']].values
subdftest=test[['Visit_Realization','Sales_Potential','Brand Awareness','Market_Size_Subcity','Customer_Potential','Sales_Total_Subcity_Q1','Sales_X_Subcity_Q1','Customer_Sales_Potential','Customer_Sales']]
Decision_Tree_Vartest = test[['Customer_Sales_Potential','Market_Size_Subcity']]

#Corellation Overall Data
visualize_corellation(subdftest)

#Application on Test Data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X_t,y_t = Decision_Tree_Vartest.values , test['Customer_Sales'].values
y_t= y_t.astype('int')
tree_t = DecisionTreeClassifier().fit(X_t, y_t)
bag_t = BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1)
bag_t.fit(X_t, y_t)
y_t_p=bag_t.predict(X_t)

visualize_classifier(bag, X_t, y_t)
visualize_classifier(DecisionTreeClassifier(), X_t, y_t)
visualize_classifier(MLPClassifier(alpha=1, max_iter=1000), X_t, y_t)

#Calculation  MSE for data and test data
Error=(y-(y_p))
MSE=sum(Error^2)/Error.shape[0]

Error_t=(y_t-(y_t_p))
MSE_t=sum(Error_t^2)/Error_t.shape[0]
mean_y=np.mean(y)
mean_y_t=np.mean(y_t)

print('Model Data MSE: ' +str(MSE.__round__()) + '\n' +'Test Data MSE: ' +str(MSE_t.__round__()))
print('Model Data Predict Error is %'+str(((np.sqrt(MSE)/mean_y)*100).round()) +  '\n'  + 'Test Data Predition Error is %' + str(((np.sqrt(MSE_t)/mean_y_t)*100).round())  )



