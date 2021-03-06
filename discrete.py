#!/usr/bin/env python

'''
ENPM 690 Spring 2020: Robot Learning
Homework 2: Discrete CMAC for 1D function

Author:
Abhinav Modi (abhi1625@umd.edu)
Graduate Student in Robotics,
University of Maryland, College Park
'''
import matplotlib.pyplot as plt
import numpy as np
import time 

n = 100
trainData = 70
testData = 30
numWeights = 35
# g = 15
lr = 0.01
testing_error = []
nEpochs = 100

dataRange = 10
X = np.arange(0,dataRange,0.1)   # start,stop,step
Y = np.cos(X)

trainingIndices = np.random.choice(np.arange(n), size = 70, replace = False).tolist()
trainingIndices.sort()
trainingData = [[X[index], Y[index]] for index in trainingIndices]

testingIndices = []

for i in np.arange(100):
	if i not in trainingIndices:
		testingIndices.append(i)

testingData = [[X[index], Y[index]] for index in testingIndices]


# Initialize Weights for Mapping
weightArr = np.zeros(35).tolist()

perEpochData = []
overlapData = []
times = []
for overlap in range(1,35):
    start_time = time.time()
    for episode in range(nEpochs):
        errors = []
        predictedOutputs = []

        for index in range(trainData):
            ip = trainingData[index][0]
            desOutput = trainingData[index][1]

            # Find association window upper and lower limits for calculating output
            associationCenter = int(numWeights*(ip/dataRange))
            if associationCenter - int(overlap/2) < 0:
                lower = 0
            else :
                lower = associationCenter - int(overlap/2)

            if associationCenter + int(overlap/2) > (numWeights-1):
                upper = numWeights-1
            else:
                upper = associationCenter + int(overlap/2)

            # Calculate output
            predOutput = 0
            for i in range(lower, upper+1):
                predOutput = predOutput + weightArr[i]*ip

            predictedOutputs.append(predOutput)

            # Calculate Error
            error = desOutput - predOutput
            errors.append(error)

            # Update weights
            for i in range(lower, upper+1):
                weightArr[i] = weightArr[i] + lr*error/(upper+1-lower)

        perEpochData.append([predictedOutputs, errors, weightArr])

    errors = []
    predictedOutputs = []
    for index in range(testData):
        ip = testingData[index][0]
        desOutput = testingData[index][1]

        # Find association window upper and lower limits for oUtput calculation
        associationCenter = int(numWeights*(ip/dataRange))
        if associationCenter - int(overlap/2) < 0:
            lower = 0
        else :
            lower = associationCenter - int(overlap/2)

        if associationCenter + int(overlap/2) > (numWeights-1):
            upper = numWeights-1
        else:
            upper = associationCenter + int(overlap/2)

        # Calculate output from Network
        predOutput = 0
        for i in range(lower, upper+1):
            predOutput = predOutput + weightArr[i]*ip

        predictedOutputs.append(predOutput)

        # Calculate Error
        error = desOutput - predOutput
        errors.append(error)
    end_time = time.time()
    time_diff = end_time - start_time
    times.append(time_diff)
    plt.plot(X,Y)
    plt.plot([testingData[index][0] for index in range(testData)], predictedOutputs)

    # plt.plot(range(nEpochs), [sum(perEpochData[index][1]) for index in range(nEpochs)])
    # plt.show()
    # mse = 0
    # for i in range(30):
    #     mse += np.linalg.norm(Y[testingIndices[i]] - predictedOutputs[i])

    # mse = mse/np.sqrt(30)    
    # overlapData.append(mse)

# # plot overlap generalization
# overlap_X = np.arange(1,35,1)
# # plt.plot(overlap_X, overlapData,'-b')
# plt.plot(overlap_X, times, '-g')
plt.show()
