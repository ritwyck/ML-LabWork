import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# loading the dataset
data = pd.read_csv('diabetes.csv')

# preparing the data frame for training classification models
xData = data.drop('class', axis=1)
yData = data['class']

# splitting the data into training and testing sets
xDataTrain, xDataTest, yDataTrain, yDataTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42)

accuracy_train = []
accuracy_test = []

# for one-level and multi-level decision trees to be calculated together

max_depth_values = [1, None]
for max_depth in max_depth_values:
    accuracy_train_depth = []
    accuracy_test_depth = []

    clf = tree.DecisionTreeClassifier(
        criterion='entropy', max_depth=max_depth)
    clf.fit(xDataTrain, yDataTrain)

    yDataTrainPrediction = clf.predict(xDataTrain)
    accuracy_train_depth.append(
        accuracy_score(yDataTrain, yDataTrainPrediction))

    yDataTestPrediction = clf.predict(xDataTest)
    accuracy_test_depth.append(accuracy_score(
        yDataTest, yDataTestPrediction))

    yDataPrediction = clf.predict(xDataTest)
    accuracyData = accuracy_score(yDataTest, yDataPrediction)

    accuracy_train.append(accuracy_train_depth)
    accuracy_test.append(accuracy_test_depth)
    print(f'Data Accuracy: {accuracyData}')
