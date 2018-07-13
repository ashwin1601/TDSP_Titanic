# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Import the data
directory = 'C:/Users/946068/Documents/Titanic'

training_dt = pd.read_csv(directory + '/train.csv')
test_dt = pd.read_csv(directory + '/test.csv')

#%%
# Calculate the survival rate of passengers for each ticket class.
alive_first = 0
alive_second = 0
alive_third = 0

for index, passenger in training_dt.iterrows():
    if passenger['Pclass'] == 1 and passenger['Survived'] == 1:
        alive_first += 1
    elif passenger['Pclass'] == 2 and passenger['Survived'] == 1:
        alive_second += 1
    elif passenger['Pclass'] == 3 and passenger['Survived'] == 1:
        alive_third += 1

survival_rate_first = alive_first / sum(training_dt.Pclass == 1)
survival_rate_second = alive_second / sum(training_dt.Pclass == 2)
survival_rate_third = alive_third / sum(training_dt.Pclass == 3)

#%%
# Plot the resulting statistic in a bar chart.
survival_rates_class = [survival_rate_first, survival_rate_second, survival_rate_third]

_ = plt.bar([1, 2, 3], survival_rates_class)
_ = plt.xlabel('Survival rate')
_ = plt.ylabel('Ticket class')
plt.show()

#%%
# Calculate the survival rate by sex.
alive_males = 0
alive_females = 0

for index, row in training_dt.iterrows():
    if row['Sex'] == 'male' and row['Survived'] == 1:
        alive_males += 1
    elif row['Sex'] == 'female' and row['Survived'] == 1:
        alive_females += 1

survival_rate_males = alive_males / sum(training_dt.Sex == 'male')
survival_rate_females = alive_females / sum(training_dt.Sex == 'female')

#%%
# Plot the resulting statistic in a bar chart.
survival_rates_sex = [survival_rate_males, survival_rate_females]

_ = plt.bar([1, 2], survival_rates_sex)
_ = plt.xlabel('Survival rate')
_ = plt.ylabel('Sex')
plt.show()

#%%
sns.distplot(training_dt.Fare)

#%%
# Investigate the effect of ticket price on the chances of surviving.
sorted_dt = training_dt.Fare.sort_values()

survival_fare_class = []

for k in range(9):
    cnt = 0
    for i in range(99):
        if training_dt.Survived[sorted_dt.index[i + 99*k]] == 1:
            cnt += 1
    survival_fare_class.append(cnt)

survival_rate_fare = [x / 99 for x in survival_fare_class]
