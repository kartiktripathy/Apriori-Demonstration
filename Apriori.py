'''Association rule learning is a rule-based machine learning method for discovering interesting relations between
variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness.'''

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''Here the dataset is of a mall customers sales , which gives us a list of what the customers bought
in their recent purchases....based on this we will find the relations between the things that has the chance of 
being grouped and picked up by the customer ....so that we can see the increase the sales of the mall.'''
#Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
'''WE write the above code to get a 2d array of the purchasing of each customer'''

#Training the Apriori Model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
'''here min_support is the no. of times u want see the product in a week divided by the total no.of transactions,here suppose we wish to see 3 times a week so =(3*7)/7501
 min_confidence is the min confidence or surety u wamnt to see in your rules
 min_lift is the quality of the rules ...below 3 its not relevant
 min_lift and max_lift is for buy one get one free offers ...depends upon specif business problems'''

#Displaying the first results coming directly from the output of the apriori function
results = list(rules)
print(results)
'''just for printing all the rules at once '''

#Putting the results well organised into a Pandas DataFrame
'''putting the rules in a tabular form using pandas library'''
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

#Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))
