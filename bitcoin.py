import statsmodels.formula.api as smf
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import math
import sys


# The path to the data folder should be given as input
if len(sys.argv) != 2:
    print('bitcoin.py <path to data folder>')
    sys.exit(1)
data_path = sys.argv[1]


# Reading the vectors from the given csv files
train1_90 = pd.read_csv(data_path+'/train1_90.csv')
train1_180 = pd.read_csv(data_path+'/train1_180.csv')
train1_360 = pd.read_csv(data_path+'/train1_360.csv')

train2_90 = pd.read_csv(data_path+'/train2_90.csv')
train2_180 = pd.read_csv(data_path+'/train2_180.csv')
train2_360 = pd.read_csv(data_path+'/train2_360.csv')

test_90 = pd.read_csv(data_path+'/test_90.csv')
test_180 = pd.read_csv(data_path+'/test_180.csv')
test_360 = pd.read_csv(data_path+'/test_360.csv')

def similarityCheck(a, b):
    #check the length of two vectors
    assert len(a) == len(b), "vectors are of different length"
    l_a = len(a)

    mean_a_vec = np.mean(a)
    std_a_vec = math.sqrt(sum([(vec_a_i - mean_a_vec)**2 for vec_a_i in a]) / (len(a)))
    #print mean_a_vec
    #print std_a_vec
    mean_b_vec = np.mean(b)
    std_b_vec = math.sqrt(sum([(vec_b_j - mean_b_vec)**2 for vec_b_j in b]) / (len(b)))

    numerator = sum([(a_k - mean_a_vec)*(b_k - mean_b_vec) for a_k, b_k in zip(a,b)])
    denominator = l_a * std_a_vec * std_b_vec

    return numerator / denominator

def computeDelta(wt, X, Xi):
    """
    This function computes equation 6 of the paper, but with the euclidean distance
    replaced by the similarity function given in Equation 9.

    Parameters
    ----------
    wt : int
        This is the constant c at the top of the right column on page 4.
    X : A row of Panda Dataframe
        Corresponds to (x, y) in Equation 6.
    Xi : Panda Dataframe
        Corresponds to a dataframe of (xi, yi) in Equation 6.

    Returns
    -------
    float
        The output of equation 6, a prediction of the average price change.
    """
    # YOUR CODE GOES HERE
    #pass
    #Implemented with several different logics, but the below one gave the desried results

    #Values from row of Panda Dataframe
    req_vals = X.values[:-1]

    this_numerator = 0
    this_denominator = 0

    #iterate through dataframe of (xi, yi)
    for xi_i in Xi.iterrows():
        new_xi_vals = xi_i[1].tolist()[:-1]
        similarity = math.exp(wt * similarityCheck(req_vals, new_xi_vals))
        this_numerator += xi_i[1]["Yi"] * similarity
        this_denominator += similarity

    return this_numerator / this_denominator



# Perform the Bayesian Regression to predict the average price change for each dataset of train2 using train1 as input.
# These will be used to estimate the coefficients (w0, w1, w2, and w3) in equation 8.
weight = 2  # This constant was not specified in the paper, but we will use 2.
trainDeltaP90 = np.empty(0)
trainDeltaP180 = np.empty(0)
trainDeltaP360 = np.empty(0)
for i in xrange(0,len(train1_90.index)) :
  trainDeltaP90 = np.append(trainDeltaP90, computeDelta(weight,train2_90.iloc[i],train1_90))
for i in xrange(0,len(train1_180.index)) :
  trainDeltaP180 = np.append(trainDeltaP180, computeDelta(weight,train2_180.iloc[i],train1_180))
for i in xrange(0,len(train1_360.index)) :
  trainDeltaP360 = np.append(trainDeltaP360, computeDelta(weight,train2_360.iloc[i],train1_360))


# Actual deltaP values for the train2 data.
trainDeltaP = np.asarray(train2_360[['Yi']])
trainDeltaP = np.reshape(trainDeltaP, -1)


# Combine all the training data
d = {'deltaP': trainDeltaP,
     'deltaP90': trainDeltaP90,
     'deltaP180': trainDeltaP180,
     'deltaP360': trainDeltaP360 }
trainData = pd.DataFrame(d)


# Feed the data: [deltaP, deltaP90, deltaP180, deltaP360] to train the linear model.
# Use the statsmodels ols function.
# Use the variable name model for your fitted model
# YOUR CODE HERE
new_model = smf.ols(formula='deltaP ~ deltaP90 + deltaP180 + deltaP360', data=trainData)

model = new_model.fit()
# Print the weights from the model
print model.params


# Perform the Bayesian Regression to predict the average price change for each dataset of test using train1 as input.
# This should be similar to above where it was computed for train2.
# YOUR CODE HERE
testDeltaP90 = np.empty(0)
testDeltaP180 = np.empty(0)
testDeltaP360 = np.empty(0)
# Constant has to be taken here, so taking const_weight = 2 as taken above
const_weight = 2

for x in xrange(0, len(train1_90.index)):
    testDeltaP90 = np.append(testDeltaP90, computeDelta(const_weight, test_90.iloc[x], train1_90))

for y in xrange(0, len(train1_180.index)):
    testDeltaP180 = np.append(testDeltaP180, computeDelta(const_weight, test_180.iloc[y], train1_180))

for z in xrange(0, len(train1_360.index)):
    testDeltaP360 = np.append(testDeltaP360, computeDelta(const_weight, test_360.iloc[z], train1_360))

# Actual deltaP values for test data.
# YOUR CODE HERE (use the right variable names so the below code works)

testDeltaP = np.empty(0)
testDeltaP = np.asarray(test_360[['Yi']])
testDeltaP = np.reshape(testDeltaP, -1)


# Combine all the test data
d = {'deltaP': testDeltaP,
     'deltaP90': testDeltaP90,
     'deltaP180': testDeltaP180,
     'deltaP360': testDeltaP360}
testData = pd.DataFrame(d)


# Predict price variation on the test data set.
result = model.predict(testData)
compare = { 'Actual': testDeltaP,
            'Predicted': result }
compareDF = pd.DataFrame(compare)


# Compute the MSE and print the result
# HINT: consider using the sm.mean_squared_error function
MSE = 0.0
# YOUR CODE HERE

#Implementing mean squared error using Sklearn metrics
MSE = sm.mean_squared_error(compareDF['Actual'], compareDF['Predicted'])
print "The MSE is %f" % (MSE)
