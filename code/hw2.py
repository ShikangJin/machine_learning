import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn import preprocessing
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer

if __name__ == "__main__":
    df = pd.read_csv("data/Training.csv")
    test = pd.read_csv("data/Test.csv")
    df_label = pd.DataFrame(data=df['response'])
    df = df.drop(columns='response')
    # print(df.dtypes)
    # numeric data: ID, response, prime, score1, score2, score3, score4, score5, contact_type, day, hour
    # categorical data: IL1, IL2, IL3, IL4, CIL1, CIL2, CIL3, CIL4, device

    # checkNull = pd.isna(df)
    # for tag in checkNull:
    #     hasMiss = False
    #     for item in checkNull[tag]:
    #         if item:
    #             hasMiss = True
    #             break
        # if not hasMiss:
            # print(tag)
    # column ID, response, prime, day, hour, device have no missing value

    # response set for each category:
    # for tag in df:
    #     if df[tag].dtype == "object":
    #         print("response set of " + tag + ": ")
    #         print(df[tag].unique())
    #         print(" ")

    # for tag in df:
    #     if df[tag].dtype == "int64" or df[tag].dtype == "float64":
    # tag = "hour"
    # df[tag].hist().get_figure().savefig("plot/" + tag + "_plot.png")

    # df["score2"].hist(bins=3).get_figure().savefig("plot/multi_plot.png")

    # score3 and score2 have a high correlation score
    # prime and score1 have a 0.25 correlation score
    # df1 = df[['score1', 'score4', 'score5',  'day', 'hour']]
    # print(df1.corr())
    #
    # var1 = 'score4'
    # var2 = 'hour'
    # X1 = df[[var1, var2]][df['response'] == 0]
    # X2 = df[[var1, var2]][df['response'] == 1]
    # plt.scatter(X1.iloc[:, 0],
    #             X1.iloc[:, 1],
    #             s=50,
    #             c='red', marker='v', label='NoResponse')
    # plt.scatter(X2.iloc[:, 0],
    #             X2.iloc[:, 1],
    #             s=50,
    #             c='blue',
    #             marker='o',
    #             label='Response')
    # plt.xlabel(var1)
    # plt.ylabel(var2)
    # plt.legend()
    # plt.grid()

    # replace null to mean value
    dfNumeric = ['score1', 'score2', 'score3', 'score4', 'score5']
    # imputer = Imputer(missing_values='NaN', strategy="mean")
    # imputer.fit(df[dfNumeric])
    # df[dfNumeric] = imputer.transform(df[dfNumeric])

    df[dfNumeric] = df[dfNumeric].fillna(value=df[dfNumeric].mean())
    test[dfNumeric] = test[dfNumeric].fillna(value=test[dfNumeric].mean())
    # dfNum = ['prime', 'day', 'hour']
    # clf = DecisionTreeClassifier()
    #
    # print(cross_val_score(clf, df[dfNum], df['response'], cv=10))
    categories = ['IL1', 'IL2', 'IL3', 'IL4', 'CIL1', 'CIL2', 'CIL3', 'CLI4', 'device', 'contact_type']
    df[categories] = df[categories].fillna(value='Not Specified')
    test[categories] = test[categories].fillna(value='Not Specified')
    # newdata = pd.get_dummies(df['contact_type'])
    # df = df.drop(columns=['contact_type']).join(newdata)
    # newdata = pd.get_dummies(test['contact_type'])
    # test = test.drop(columns=['contact_type']).join(newdata)
    for tag in categories:
        features = pd.concat([df[tag], test[tag]]).unique().tolist()
        df[tag] = df[tag].astype('category').cat.set_categories(features)
        test[tag] = test[tag].astype('category').cat.set_categories(features)
        # print(df[tag])
    newdata = pd.get_dummies(df[categories])
    df = df.drop(columns=categories).join(newdata)
    newdata = pd.get_dummies(test[categories])
    test = test.drop(columns=categories).join(newdata)
    # print(df.shape) #index not found error
    # print(newdata.shape)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(df)
    X_train, X_test, y_train, y_test = train_test_split(df, df_label, test_size=0.2, random_state=1)
    classifier = LogisticRegression(C=0.00001)
    classifier.fit(X_train, y_train.values.ravel())
    y_pred = classifier.predict(X_test)
    y_test = y_test.values.reshape(1, 1363)[0]
    score_micro = f1_score(y_test, y_pred, average='micro')
    score_macro = f1_score(y_test, y_pred, average='macro')

    print(score_micro)
    print(score_macro)