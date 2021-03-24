import numpy as np

b = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
# clss = [0, 0, 1, 1]
# idxs = [i for i, c in enumerate(clss) if c == 0]
# print([b[x] for x in idxs])

b1, b2, b3 = b[:-1]
print(b1)

# for i, item in enumerate(b):
#   item.append(i)
b = np.array(b)
index = np.array([i for i in range(len(b))]).reshape(len(b), 1)
b= np.hstack((b, index))

# a = np.array([1999, 999, 999])
print(b[int(np.argwhere(b[:, -1]==2)), :-1])

# for i in range(len(b)):
#     b[i][:-2] = np.add(b[i][:-2], a)
# print(np.append(a, 1))