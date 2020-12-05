import math
import numpy as np


###### question1 #####
def PxY1(x):
    pxy1 = (1 / (4 * math.sqrt(2 * math.pi))) * math.exp(-(math.pow((x - 4), 2)) / (2 * 16))
    return pxy1


def PxY2(x):
    pxy2 = (1 / (4 * math.sqrt(2 * math.pi))) * math.exp(-(math.pow((x - 10), 2)) / (2 * 16))
    return pxy2


result = (0.5 * PxY1(16)) / (0.5 * PxY1(16) + 0.5 * PxY2(16))

result2 = (0.5 * PxY2(16)) / (0.5 * PxY2(16) + 0.5 * PxY1(16))

p = [0, 2, 10, 12, 14, 16]
sumNumerator2 = 0.0
sumNumerator = 0.0
sumDominator2 = 0.0
sumDominator = 0.0

for ele in p:
    sumNumerator2 += (0.5 * PxY1(ele)) / (0.5 * PxY1(ele) + 0.5 * PxY2(ele)) * math.pow((ele - 3.18), 2)
    sumDominator2 += (0.5 * PxY1(ele)) / (0.5 * PxY1(ele) + 0.5 * PxY2(ele))

    sumNumerator += ((0.5 * PxY2(ele)) / (0.5 * PxY2(ele) + 0.5 * PxY1(ele))) * math.pow((ele - 12.56), 2)
    sumDominator += (0.5 * PxY2(ele)) / (0.5 * PxY2(ele) + 0.5 * PxY1(ele))







