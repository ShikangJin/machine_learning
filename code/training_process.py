import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

if __name__ == "__main__":
    df = pd.read_csv("data/Training.csv")
    test = pd.read_csv("data/Test.csv")
    df_label = pd.DataFrame(data=df['response'])
    X_train, X_test, y_train, y_test = train_test_split(df, df_label, test_size=0.2, random_state=1)

    dfNumeric = ['score1', 'score2', 'score3', 'score4', 'score5']
    X_train.loc[:, dfNumeric] = X_train[dfNumeric].fillna(value=X_train[dfNumeric].mean())
    X_test.loc[:, dfNumeric] = X_test[dfNumeric].fillna(value=X_test[dfNumeric].mean())

    categories = ['IL1', 'IL2', 'IL3', 'IL4', 'CIL1', 'CIL2', 'CIL3', 'CLI4', 'contact_type']
    X_train.loc[:, categories] = X_train[categories].fillna(value='Not Specified')
    X_test.loc[:, categories] = X_test[categories].fillna(value='Not Specified')

    categories = ['IL1', 'IL2', 'IL3', 'IL4', 'CIL1', 'CIL2', 'CIL3', 'CLI4', 'device', 'contact_type']
    for tag in categories:
        features = pd.concat([X_train[tag], X_test[tag]]).unique().tolist()
        X_train.loc[:, tag] = X_train[tag].astype('category').cat.set_categories(features)
        X_test.loc[:, tag] = X_test[tag].astype('category').cat.set_categories(features)
    train_OneHot = pd.get_dummies(X_train)
    test_OneHot = pd.get_dummies(X_test)

    featureList = train_OneHot.columns.drop(['response', 'ID', 'score2', 'score3']).tolist()
    X_train = train_OneHot[featureList]
    y_train = train_OneHot['response']
    X_test = test_OneHot[featureList]
    y_test = test_OneHot['response']

    scaler = StandardScaler(with_mean=False)
    classifier = DecisionTreeClassifier()
    classifier2 = LogisticRegression()
    pca = PCA(n_components=5)

    # Prediction
    n_components = [5, 10, 20]
    class_weight = [{0: 1, 1: 1}, {0: 1, 1: 4}, {0: 1, 1: 16}, {0: 1, 1: 64}]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    C = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    boost = BaggingClassifier(classifier2)
    pip = Pipeline([('scaler', scaler), ('pca', pca), ('boost', boost)])

    # f1_scorer = make_scorer(f1_score)
    # print(cross_val_score(pip, X_train, y_train, scoring='f1', cv=5).mean())

    estimator = GridSearchCV(pip,
                             dict(pca__n_components=n_components,
                                  boost__base_estimator__class_weight=class_weight,
                                  boost__base_estimator__C=C),
                             scoring='f1')

    estimator.fit(X_train, y_train)
    # 'pca__n_components': 10, 'clf__class_weight': {0: 1, 1: 1}
    #
    # boost = boost.set_params(pca__n_components=20, clf__class_weight={0: 1, 1: 64},
    #                          clf__C=1e-05)

    print(estimator.best_params_)
    # pip.fit(X_train, y_train)
    print(cross_val_score(estimator.best_estimator_, X_test, y_test, cv=5, scoring='f1').mean())

    print(estimator.best_score_)

    # cv = cross_val_score(pip, X_train, y_train, scoring=f1_scorer)
    # # print(cross_val_score(pip, X_train, y_train, scoring=f1_scorer).mean())
    # # print(classifier.feature_importances_)
    #
    # plt.figure()
    # plt.xlabel("Training examples")
    # plt.ylabel("Score")
    #
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    # train_sizes, train_scores, test_scores = learning_curve(
    #     estimator, X_train, y_train, train_sizes=np.linspace(.1, 1.0, 5))
    #
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # plt.grid()
    #
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
    #          label="Training score")
    # plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
    #          label="Cross-validation score")
    #
    # plt.legend(loc="best")
    # plt.show()
    #



    # pip.fit(X_train, y_train.values.ravel())
    # y_pred = pip.predict(X_test)
    # y_test = y_test.values.reshape(1, 1363)[0]
    # score_micro = precision_score(y_test, y_pred, average='micro')
    # score_macro = precision_score(y_test, y_pred, average='macro')
    # print(score_micro)
    # print(score_macro)



