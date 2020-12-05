import math
import numpy as np


###### question3 #####

def P_color_p(t, color, arr4):
    totalNum = 0
    num = 0
    for i in range(0, np.size(arr4, 0)):
        if arr4[i][0] > t:
            totalNum += 1
            if arr4[i][1] == color:
                num += 1
    result = num / totalNum
    # print(result)
    return result


def P_color_pp(t, color, arr4):
    totalNum = 0
    num = 0
    for i in range(0, np.size(arr4, 0)):
        if arr4[i][0] <= t:
            totalNum += 1
            if arr4[i][1] == color:
                num += 1
    result = num / totalNum
    # print(result)
    return result


# 3.2 x[1] => 2.5
arrQuestion3 = np.array([[4, 1], [5, 1], [6, 1], [7, 1], [4, 2], [2, 2], [3, 2], [6, 2]])
threshHold = np.array([2.5, 3.5, 4.5, 5.5, 6.5])
minThresh = 10000.0
resultThresh = 10000.0
for index in range(0, 5):
    thresh = threshHold[index]
    sum1 = P_color_p(thresh, 1, arrQuestion3) * (1 - P_color_p(thresh, 1, arrQuestion3))
    sum2 = P_color_pp(thresh, 1, arrQuestion3) * (1 - P_color_pp(thresh, 1, arrQuestion3))
    sum3 = P_color_p(thresh, 2, arrQuestion3) * (1 - P_color_p(thresh, 2, arrQuestion3))
    sum4 = P_color_pp(thresh, 2, arrQuestion3) * (1 - P_color_pp(thresh, 2, arrQuestion3))
    totalp = sum1 + sum3
    totalpp = sum2 + sum4
    # print("============")
    # print(index)
    # print(totalp)
    # print(totalpp)
    # print("============")
    if (totalp <= totalpp) and (totalpp < minThresh):
        minThresh = totalpp
        resultThresh = thresh
    elif (totalp > totalpp) and (totalp < minThresh):
        minThresh = totalp
        resultThresh = thresh
print("Question3.2 determine x[1]:")
print(resultThresh)
# 3.2 x[2] on the left
pointsOnLeft = np.array([[4, 2], [2, 2],[4,1],[3,2]])
pointsOnRight = np.array([ [5, 1], [3, 1], [1, 2], [3, 1], [2, 2]])
threshHold_y = np.array([1.5, 2.5, 3.5, 4.5])
threshHold_yLeft = np.array([2.5, 3.5])
minThresh = 10000.0
resultThresh = 0.0
for index in range(0, 2):
    thresh = threshHold_yLeft[index]
    sum1 = P_color_p(thresh, 1, pointsOnLeft) * (1 - P_color_p(thresh, 1, pointsOnLeft))
    sum2 = P_color_pp(thresh, 1, pointsOnLeft) * (1 - P_color_pp(thresh, 1, pointsOnLeft))
    sum3 = P_color_p(thresh, 2, pointsOnLeft) * (1 - P_color_p(thresh, 2, pointsOnLeft))
    sum4 = P_color_pp(thresh, 2, pointsOnLeft) * (1 - P_color_pp(thresh, 2, pointsOnLeft))
    totalp = sum1 + sum3
    totalpp = sum2 + sum4
    #print(totalp)
    #print(totalpp)
    if (totalp < totalpp) and (totalpp < minThresh):
        minThresh = totalpp
        resultThresh = thresh
    elif (totalp > totalpp) and (totalp < minThresh):
        minThresh = totalp
        resultThresh = thresh
print("Question3.2 determine x[2] on the left:")
print(resultThresh)
# 3.2 x[2] on the right
pointsOnRight = np.array([ [5, 1], [3, 1], [1, 2], [3, 1], [2, 2]])
threshHold_y = np.array([1.5, 2.5, 3.5, 4.5])
minThresh = 10000.0
resultThresh = 0.0
for index in range(0, 4):
    thresh = threshHold_y[index]
    sum1 = P_color_p(thresh, 1, pointsOnRight) * (1 - P_color_p(thresh, 1, pointsOnRight))
    sum2 = P_color_pp(thresh, 1, pointsOnRight) * (1 - P_color_pp(thresh, 1, pointsOnRight))
    sum3 = P_color_p(thresh, 2, pointsOnRight) * (1 - P_color_p(thresh, 2, pointsOnRight))
    sum4 = P_color_pp(thresh, 2, pointsOnRight) * (1 - P_color_pp(thresh, 2, pointsOnRight))
    totalp = sum1 + sum3
    totalpp = sum2 + sum4
    #print(totalp)
    #print(totalpp)
    if (totalp < totalpp) and (totalpp < minThresh):
        minThresh = totalpp
        resultThresh = thresh
    elif (totalp > totalpp) and (totalp < minThresh):
        minThresh = totalp
        resultThresh = thresh
print("Question3.2 determine x[2] on the right:")
print(resultThresh)
