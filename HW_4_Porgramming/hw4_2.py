import math
import numpy as np

###### question2 #####

arr = np.array([[4, 4], [5, 5], [6, 3], [7, 3], [4, 3], [2, 4], [3, 2], [6, 1]])
correct = True

for i in range(0, 8):
    indexOfi = 0
    indexOfj = 0
    distance = 1000000.0
    for j in range(0, 8):
        if i != j:
            dis = math.sqrt(math.pow((arr[i][0] - arr[j][0]), 2) + math.pow((arr[i][1] - arr[j][1]), 2))
            if dis < distance:
                distance = dis
                indexOfj = j
                indexOfi = i
    print("Qusetion 2.1:")
    if ((indexOfi < 4) and (indexOfj < 4)) or ((indexOfi >= 4) and (indexOfj >= 4)):
        print(True)
    else:
        print(False)

#####2.2
# if i is in range(0,4) it is blue
# if i is in range(4,8) it is red

for i in range(0, 8):
    arr2 = np.array([[0, 0.0], [1, 0.0], [2, 0.0], [3, 0.0], [4, 0.0], [5, 0.0], [6, 0.0], [7, 0.0]])
    for j in range(0, 8):
        dis = 0.0
        dis = math.sqrt(math.pow((arr[i][0] - arr[j][0]), 2) + math.pow((arr[i][1] - arr[j][1]), 2))
        arr2[j][0] = j
        arr2[j][1] = dis
    #print(arr2)
    min1 = 10000.0
    indexOfMin1 = 0
    for k in range(0, 8):
        if (arr2[k][1] != 0) and arr2[k][1] < min1:
            min1 = arr2[k][1]
            indexOfMin1 = k
    np.delete(arr2, indexOfMin1, axis=0)
    min2 = 10000.0
    indexOfMin2 = 0
    for k in range(0, 7):
        if (arr2[k][1] != 0) and arr2[k][1] < min2:
            min2 = arr2[k][1]
            indexOfMin2 = k
    np.delete(arr2, indexOfMin2, axis=0)
    min3 = 10000.0
    indexOfMin3 = 0
    for k in range(0, 6):
        if (arr2[k][1] != 0) and arr2[k][1] < min3:
            min3 = arr2[k][1]
            indexOfMin3 = k
    arr3 = np.array([indexOfMin1, indexOfMin2, indexOfMin3])
    red = 0
    blue = 0
    color = 0
    if i < 4:
        #0 is blue
        color = 0
    else:
        #1 is red
        color = 1
    for k in range(0, 3):
        if arr3[k] < 4:
            blue += 1
        else:
            red += 1
    print("Qusetion 2.2:")
    if ((red > blue) and color == 1) or ((red < blue) and color == 0):
        print(True)
    else:
        print(False)
