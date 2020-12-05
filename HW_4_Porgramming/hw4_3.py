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
    #print(result)
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
    #print(result)
    return result


# 3.1 thresh for x[1] => thresh = 3.5
# [x,y] => x is value of x[1]; y is color (1 is blue; 2 is red)
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
    total = sum1 + sum2 + sum3 + sum4
    #print(total)
    if total < minThresh:
        minThresh = total
        resultThresh = thresh
print("Question3.1 determine x[1]:")
print(resultThresh)

# 3.1 thresh for x[2]left => 2.5; for x[2]right => 1.5

pointsOnRight = np.array([[4, 1], [5, 1], [3, 1], [3, 2], [1, 2], [3, 1]])
threshHold_y = np.array([1.5, 2.5, 3.5, 4.5])
minThresh = 10000.0
resultThresh = 10000.0
# Question3.1 determine x[1] on the right
for index in range(0, 4):
    thresh = threshHold_y[index]
    sum1 = float(P_color_p(thresh, 1, pointsOnRight) * (1 - P_color_p(thresh, 1, pointsOnRight)))
    sum2 = float(P_color_pp(thresh, 1, pointsOnRight) * (1 - P_color_pp(thresh, 1, pointsOnRight)))
    sum3 = float(P_color_p(thresh, 2, pointsOnRight) * (1 - P_color_p(thresh, 2, pointsOnRight)))
    sum4 = float(P_color_pp(thresh, 2, pointsOnRight) * (1 - P_color_pp(thresh, 2, pointsOnRight)))
    total = float(sum1 + sum2 + sum3 + sum4)
    #print(total)
    if total < minThresh:
        minThresh = total
        resultThresh = thresh
print("Question3.1 determine x[1] on the right:")
print(resultThresh)
# Question3.1 determine x[1] on the left
threshHold_yLeft = np.array([2.5, 3.5])
pointsOnLeft = np.array([[4, 2], [2, 2]])
minThresh = 10000.0
resultThresh = 10000.0
for index in range(0, 2):
    thresh = threshHold_yLeft[index]
    #print(thresh)
    sum1 = P_color_p(thresh, 1, pointsOnLeft) * (1 - P_color_p(thresh, 1, pointsOnLeft))
    sum2 = P_color_pp(thresh, 1, pointsOnLeft) * (1 - P_color_pp(thresh, 1, pointsOnLeft))
    sum3 = P_color_p(thresh, 2, pointsOnLeft) * (1 - P_color_p(thresh, 2, pointsOnLeft))
    sum4 = P_color_pp(thresh, 2, pointsOnLeft) * (1 - P_color_pp(thresh, 2, pointsOnLeft))
    total = sum1 + sum2 + sum3 + sum4
    if total < minThresh:
        minThresh = total
        resultThresh = thresh
    #print(total)
print("Question3.1 determine x[1] on the left:")
print(resultThresh)
# 3.2 x[1] => 2.5
# minThresh = 10000.0
# resultThresh = 10000.0
# for index in range(0, 5):
#     thresh = threshHold[index]
#     sum1 = P_color_p(thresh, 1, arrQuestion3) * (1 - P_color_p(thresh, 1, arrQuestion3))
#     sum2 = P_color_pp(thresh, 1, arrQuestion3) * (1 - P_color_pp(thresh, 1, arrQuestion3))
#     sum3 = P_color_p(thresh, 2, arrQuestion3) * (1 - P_color_p(thresh, 2, arrQuestion3))
#     sum4 = P_color_pp(thresh, 2, arrQuestion3) * (1 - P_color_pp(thresh, 2, arrQuestion3))
#     totalp = sum1 + sum3
#     totalpp = sum2 + sum4
#     if (totalp < totalpp) and (totalpp < minThresh):
#         minThresh = totalpp
#         resultThresh = thresh
#     elif (totalp > totalpp) and (totalp < minThresh):
#         minThresh = totalpp
#         resultThresh = thresh
# print(resultThresh)
######3.2 left and rightx[2]=>1.5
# pointsOnLeft = np.array([[4, 2]])
# pointsOnRight = np.array([[4, 1], [5, 1], [3, 1], [3, 2], [1, 2], [3, 1], [2, 2]])
# minThresh = 10000.0
# resultThresh = 10000.0
# for index in range(0, 4):
#     thresh = threshHold_y[index]
#     sum1 = P_color_p(thresh, 1, pointsOnLeft) * (1 - P_color_p(thresh, 1, pointsOnLeft))
#     sum2 = P_color_pp(thresh, 1, pointsOnLeft) * (1 - P_color_pp(thresh, 1, pointsOnLeft))
#     sum3 = P_color_p(thresh, 2, pointsOnLeft) * (1 - P_color_p(thresh, 2, pointsOnLeft))
#     sum4 = P_color_pp(thresh, 2, pointsOnLeft) * (1 - P_color_pp(thresh, 2, pointsOnLeft))
#     totalp = sum1 + sum3
#     totalpp = sum2 + sum4
#     if (totalp < totalpp) and (totalpp < minThresh):
#         minThresh = totalpp
#         resultThresh = thresh
#     elif (totalp > totalpp) and (totalp < minThresh):
#         minThresh = totalpp
#         resultThresh = thresh
# print(resultThresh)
