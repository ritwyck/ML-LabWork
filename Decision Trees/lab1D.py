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

# defining the range of values for max_depth
min_samples_leaf_values = range(1, 768, 5)

accuracy_train = []
accuracy_test = []

# for one-level and multi-level decision trees to be calculated together
max_depth_values = [1, None]
for max_depth in max_depth_values:
    accuracy_train_depth = []
    accuracy_test_depth = []

    for min_samples_leaf in min_samples_leaf_values:
        clf = tree.DecisionTreeClassifier(
            criterion='entropy', max_depth=max_depth, min_samples_leaf=min_samples_leaf)
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

# plotting the results
for i, max_depth in enumerate(max_depth_values):
    plt.figure(figsize=(7, 7))
    plt.plot(
        min_samples_leaf_values, accuracy_train[i], label=f'Training (max_depth={max_depth})', linestyle='--')
    plt.plot(
        min_samples_leaf_values, accuracy_test[i], label=f'Test (max_depth={max_depth})')

    plt.xlabel('min_samples_leaf')
    plt.ylabel('Accuracy')
    plt.title(
        f'Accuracy Rates for max_depth={max_depth} and Different min_samples_leaf Values')
    plt.legend()
    plt.show()
