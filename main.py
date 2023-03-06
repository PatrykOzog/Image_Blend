import timeit
code_to_time = """

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import timeit


def scale_image(image, scale):
    rows, cols = len(image), len(image[0])
    new_rows, new_cols = int(rows * scale), int(cols * scale)
    new_image = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
    for row in range(new_rows):
        for col in range(new_cols):
            original_row, original_col = row / scale, col / scale
            row_ratio, col_ratio = original_row - int(original_row), original_col - int(original_col)
            top_left = image[int(original_row)][int(original_col)]
            top_right = image[int(original_row)][min(int(original_col) + 1, cols-1)]
            bottom_left = image[min(int(original_row) + 1, rows-1)][int(original_col)]
            bottom_right = image[min(int(original_row) + 1, rows-1)][min(int(original_col) + 1, cols-1)]
            new_image[row][col] = (1 - row_ratio) * (1 - col_ratio) * top_left + (1 - row_ratio) * col_ratio * top_right + row_ratio * (1 - col_ratio) * bottom_left + row_ratio * col_ratio * bottom_right
    return new_image

A = cv.imread('Alice_(apple).jpg')
file_path = 'Orange_ungeschältl.jpeg'

with open(file_path, "rb") as f:
    img_array = f.read()
B = cv.imdecode(np.frombuffer(img_array, np.uint8), -1)

A = cv.cvtColor(A, cv.COLOR_BGR2RGB)
B = cv.cvtColor(B, cv.COLOR_BGR2RGB)

A = scale_image(A, 1024/1028)

G = [row[:] for row in A]
gpA = [G]
for i in range(6):
    G = scale_image(G, 0.5)
    gpA.append(G)

G = [row[:] for row in B]
gpB = [G]
for i in range(6):
    G = scale_image(G, 0.5)
    gpB.append(G)

lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = scale_image(gpA[i], 2)
    L = [[gpA[i-1][j][k]-GE[j][k] for k in range(len(gpA[i-1][0]))] for j in range(len(gpA[i-1]))]
    lpA.append(L)

lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = scale_image(gpB[i], 2)
    L = [[gpB[i-1][j][k]-GE[j][k] for k in range(len(gpB[i-1][0]))] for j in range(len(gpB[i-1]))]
    lpB.append(L)

LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = len(la), len(la[0]), len(la[0][0])
    ls = [[la[i][j] for j in range(cols//2)] + [lb[i][j] for j in range(cols//2, cols)] for i in range(rows)]
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 6):
    ls_ = scale_image(ls_, 2)
    for j in range(len(ls_)):
        for k in range(len(ls_[j])):
            ls_[j][k] += LS[i][j][k]

real = [[A[i][j] for j in range(cols//2)] + [B[i][j] for j in range(cols//2, cols)] for i in range(rows)]
ls_ = [[x/255 for x in row] for row in ls_]
real = [[x/255 for x in row] for row in real]

plt.imshow(ls_)
plt.savefig("blend3.jpg")
plt.imshow(real)
plt.savefig("blend4.jpg")

"""

execution_time = timeit.timeit(code_to_time, number=10)
print(f'Czas średniego wykonania: {execution_time/10} sekund')

# plt.imshow(ls_)
# plt.show()
# plt.imshow(real)
# plt.show()