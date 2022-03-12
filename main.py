# -*- coding: utf-8 -*-
"""

1st Part: Glass Material Classification
2nd Part: Concrete Material Strength Estimation from Data

@author: Umut Kalman
"""

import numpy as np
import pandas as pd
import math
from collections import Counter

glass_df = pd.read_csv("glass.csv")
concrete_df = pd.read_csv("Concrete_Data_Yeh.csv")

glass_arr = glass_df.to_numpy()
concrete_arr = concrete_df.to_numpy()


# Method for feature normalizing data
def f_normalization(a_five_fold_arr):
    normalized_arr = np.array(
        [np.zeros([a_five_fold_arr[0].shape[0], a_five_fold_arr[0].shape[1]]),
         np.zeros([a_five_fold_arr[1].shape[0], a_five_fold_arr[1].shape[1]]),
         np.zeros([a_five_fold_arr[2].shape[0], a_five_fold_arr[2].shape[1]]),
         np.zeros([a_five_fold_arr[3].shape[0], a_five_fold_arr[3].shape[1]]),
         np.zeros([a_five_fold_arr[4].shape[0], a_five_fold_arr[4].shape[1]])],
        dtype=object)

    for k in range(len(a_five_fold_arr)):
        for i in range(a_five_fold_arr[k].shape[1] - 1):
            max_attr = np.max(a_five_fold_arr[k][:, i])
            min_attr = np.min(a_five_fold_arr[k][:, i])

            for j in range(len(a_five_fold_arr[k][:, i])):
                normalized_arr[k][j][i] = (a_five_fold_arr[k][j][i] - min_attr) / (max_attr - min_attr)

    for i in range(len(a_five_fold_arr)):
        for j in range(len(a_five_fold_arr[i])):
            normalized_arr[i][j][a_five_fold_arr[0].shape[1] - 1] = a_five_fold_arr[i][j][a_five_fold_arr[0].shape[1] - 1]

    return normalized_arr


# Method for checking if all 5 splitted data contains at least one of each sample of classes
def class_control(class_column):
    check_list = [1, 2, 3, 5, 6, 7]

    value = True
    for i in check_list:
        if i not in class_column:
            value = False
            break

    return value


# Method for shuffling the whole glass data array and dividing it to 5
def glass_five_fold(base_array):
    five_fold_arr = np.array(
        [np.zeros([43, 10]), np.zeros([43, 10]), np.zeros([43, 10]), np.zeros([43, 10]), np.zeros([42, 10])],
        dtype=object)

    while True:
        boolean_value = True
        np.random.shuffle(base_array)

        first_increment, second_increment = 0, 43

        for i in range(5):

            if i == 4:
                five_fold_arr[i] = base_array[172:214]

            else:
                five_fold_arr[i] = base_array[first_increment:second_increment]
                first_increment += 43
                second_increment += 43

            if not class_control(five_fold_arr[i][:, 9]):
                boolean_value = False
                break

        if not boolean_value:
            continue

        break

    return five_fold_arr


# Method for shuffling the whole concrete data array and dividing it to 5
def concrete_five_fold(base_array):
    five_fold_arr = np.array(
        [np.zeros([206, 9]), np.zeros([206, 9]), np.zeros([206, 9]), np.zeros([206, 9]), np.zeros([206, 9])],
        dtype=object)

    np.random.shuffle(base_array)

    first_increment, second_increment = 0, 206

    for i in range(5):
        five_fold_arr[i] = base_array[first_increment:second_increment]
        first_increment += 206
        second_increment += 206

    return five_fold_arr


class KNN:

    def __init__(self, k):
        self.k = k

    # Training the model-actually setting attribute and class train arrays-
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Calculating distance by using Euclidian distance method
    def distance(self, X1, X2):
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(X1, X2)]))
        return distance

    # Most important method in KNN class
    # Takes boolean value $isWeighted as an argument(if weighted is True weighted KNN occurs else standard)
    # Predicts class of glass or estimates concrete strength
    def predict(self, x_test, isWeighted):

        final_output = []
        for i in range(len(x_test)):
            dist_list = []
            votes = []

            # Calculating distances
            for j in range(len(self.X_train)):
                dist = self.distance(self.X_train[j], x_test[i])
                dist_list.append(dist)

            min_dist_index_list = []
            copy_list = dist_list[:]

            # Finding k nearest neighbors
            for x in range(self.k):

                min_index = dist_list.index(min(copy_list))

                if min_index not in min_dist_index_list:
                    min_dist_index_list.append(min_index)

                else:
                    min_index = dist_list[min_dist_index_list[-1] + 1:].index(min(copy_list)) + min_dist_index_list[
                        -1] + 1

                    min_dist_index_list.append(min_index)

                copy_list.remove(min(copy_list))

            # Prediction/Estimation with weighted KNN
            if (isWeighted):

                # Calculating weights
                weight_and_y = []
                for index in min_dist_index_list:
                    if (dist_list[index] == 0):
                        # if distance between test data and train data is 0, a great integer value is added to weights that represents the same data
                        weight_and_y.append([100000000000, self.y_train[index]])
                        continue

                    weight_and_y.append([1 / dist_list[index], self.y_train[index]])

                # Predicting classes for glass data
                if (x_test.shape[0] == 43 or x_test.shape[0] == 42):

                    weight_sums = []
                    weight_sums.append(weight_and_y[0])

                    for i in range(1, len(weight_and_y)):
                        is_added = False
                        for j in range(0, len(weight_sums)):
                            if weight_and_y[i][1] == weight_sums[j][1]:
                                is_added = True
                                break

                        if (is_added):
                            weight_sums[j][0] = weight_sums[j][0] + weight_and_y[i][0]

                        else:
                            weight_sums.append(weight_and_y[i])

                    maximum = weight_sums[0]
                    for i in range(len(weight_sums)):
                        if weight_sums[i][0] > maximum[0]:
                            maximum = weight_sums[i]

                    final_output.append(maximum[1])

                # Estimating concrete strength for concrete data
                elif (x_test.shape[0] == 206):
                    total = 0
                    weight_total = 0
                    for i in range(len(weight_and_y)):
                        total += (weight_and_y[i][0] * weight_and_y[i][1])
                        weight_total += weight_and_y[i][0]

                    final_output.append(total / weight_total)

            # Prediction/Estimation without weighted KNN
            else:

                for index in min_dist_index_list:
                    votes.append(self.y_train[index])

                # Prediction for glass test data
                if (x_test.shape[0] == 43 or x_test.shape[0] == 42):

                    occurence_count = Counter(votes)
                    result = occurence_count.most_common(1)[0][0]
                    final_output.append(result)

                # Estimation for concrete test data
                elif (x_test.shape[0] == 206):
                    final_output.append(sum(votes) / self.k)

        return final_output

    # Accuracy calculation of glass data KNN(Returning a number between 0 and 100 that represents how true the model predicted)
    def accuracy(self, x_test, y_test, isWeighted):
        predictions = self.predict(x_test, isWeighted)
        print("k is: ", self.k, end=" ")

        return (predictions == y_test).sum() / len(y_test) * 100

    # Calculating Mean Absolute Error of Regression Data's KNN
    def mean_absolute_error(self, x_test, y_test, isWeighted):

        predictions = self.predict(x_test, isWeighted)
        sums = 0

        for i in range(len(y_test)):
            sums += abs(y_test[i] - predictions[i])

        print("k is: ", self.k, end=" ")
        return sums / len(y_test)


# Main method for executing KNN operations
def execute_knn(base_array, param_array):
    cols = param_array[0].shape[1]

    # Setting 5-fold data's 4 arrays as train and remaining array as test
    for i in range(len(param_array)):

        print("Slice ", i)

        attribute_test = param_array[i][:, :cols - 1]
        class_test = param_array[i][:, cols - 1]
        attribute_train = np.zeros([base_array.shape[0] - len(attribute_test), cols - 1])
        class_train = np.zeros(base_array.shape[0] - len(class_test))

        index = 0
        for j in range(len(param_array)):
            if j is not i:
                for x in range(len(param_array[j])):
                    attribute_train[index] = param_array[j][x, :cols - 1]
                    class_train[index] = param_array[j][x][cols - 1]
                    index += 1

        k_values = [1, 3, 5, 7, 9]

        average_accuracy = 0
        average_mae = 0

        # Without Weighting Executing KNN and printing accuracy/mean absolute error
        print("WITHOUT WEIGHTING")
        for k_value in k_values:

            knn = KNN(k_value)
            knn.fit(attribute_train, class_train)
            if (base_array.shape[0] == 214):
                accuracy = knn.accuracy(attribute_test, class_test, False)
                average_accuracy += accuracy
                print("Accuracy: ", accuracy)

            elif (base_array.shape[0] == 1030):
                mae = knn.mean_absolute_error(attribute_test, class_test, False)
                average_mae += mae
                print("Mean Absolute Error: ", mae)

        if (base_array.shape[0] == 214):
            average_accuracy /= 5
            print("\nAverage Accuracy of Slice ", i, ": ", average_accuracy)

        elif (base_array.shape[0] == 1030):
            average_mae /= 5
            print("\nAverage Mean Absolute Error of Slice ", i, ": ", average_mae)

        print("\n::::::::::::::::::::::::::::::::::::::\n")

        average_accuracy = 0
        average_mae = 0
        # With Weighting Executing KNN and printing accuracy/mean absolute error
        print("WITH WEIGHTING")
        for k_value_2 in k_values:

            knn = KNN(k_value_2)
            knn.fit(attribute_train, class_train)

            if (base_array.shape[0] == 214):
                accuracy = knn.accuracy(attribute_test, class_test, True)
                average_accuracy += accuracy
                print("Accuracy: ", accuracy)

            elif (base_array.shape[0] == 1030):
                mae = knn.mean_absolute_error(attribute_test, class_test, True)
                average_mae += mae
                print("Mean Absolute Error: ", mae)

        if (base_array.shape[0] == 214):
            average_accuracy /= 5
            print("\nAverage Accuracy of Slice ", i, ": ", average_accuracy)

        elif (base_array.shape[0] == 1030):
            average_mae /= 5
            print("\nAverage Mean Absolute Error of Slice ", i, ": ", average_mae)

        print("\n**************************************\n")

    print("-----------------------------------------------------")
    print("#####################################################")


# 5-fold cross validation
glass_five_fold_arr = glass_five_fold(glass_arr)
normalized_glass_arr = f_normalization(glass_five_fold_arr)

concrete_five_fold_arr = concrete_five_fold(concrete_arr)
normalized_concrete_arr = f_normalization(concrete_five_fold_arr)

print("WITHOUT NORMALIZATION GLASS DATA: ")
execute_knn(glass_arr, glass_five_fold_arr)

print("WITH NORMALIZATION GLASS DATA: ")
execute_knn(glass_arr, normalized_glass_arr)

print("\n\n\n\n")

print("WITHOUT NORMALIZATION CONCRETE DATA: ")
execute_knn(concrete_arr, concrete_five_fold_arr)

print("WITH NORMALIZATION CONCRETE DATA: ")
execute_knn(concrete_arr, normalized_concrete_arr)
