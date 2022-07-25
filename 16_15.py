#/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

grad = [[0.1, 0.4, 0.2, -0.2],
        [20.0, 5.0, 12.0, -5.0]]
eps = 1
beta1, beta2 = 0.9, 0.99
res = [[], []]
avg = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
unbias_avg = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
sqr_avg = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
print(f"eps = {eps}, beta1, beta2 = {beta1}, {beta2}")
for i in range(2):
    for j in range(len(grad[i])):
        avg[i][j] = round(beta1 * avg[i][j-1] + (1-beta1) * grad[i][j], 4)
        unbias_avg[i][j] = round(avg[i][j] / (1 - (beta1**(j+1))), 4)
        sqr_avg[i][j] = round(beta2 * sqr_avg[i][j-1] + (1 - beta2) * (grad[i][j]**2), 4)
        res[i].append(round(3e-1 * unbias_avg[i][j] / math.sqrt(sqr_avg[i][j] + eps), 4))
    print(f"{i + 1}: grads = {grad[i]}, avg = {avg[i]},\n unbias_avg  = {unbias_avg[i]}"
          f"sqr_avg = {sqr_avg[i]},\nres = {res[i]}\n\n")


