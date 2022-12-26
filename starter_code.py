# importing modules
from samples import *
import numpy as np
import matplotlib.pyplot as plt


# function for visualizing the data
def visualize(x, y, n, d1, d2):
    for i in range(n):
        label = y[i]
        pixels = x[i]
        pixels = np.array(pixels, dtype='uint8')
        pixels = pixels.reshape(d1, d2)
        ax = plt.subplot(1, n, i + 1)
        ax = plt.imshow(pixels, cmap='gray')
        plt.title('{label}'.format(label=label))
        plt.plot()
    plt.show()


def read_labels(filepath):
    arr = []
    with open(filepath, "r", encoding='utf-8') as file:
        for line in file:
            line = line.rstrip("\n")
            arr.append(line)
    barr = [int(i) for i in arr]
    arr = np.array(barr)
    return arr


# 5000x28x28
def read_lines(filepath, n):
    arr = []
    result = []
    training_digits = []
    row_count = 0
    with open(filepath, 'r', encoding='utf-8') as file:
        # for each 28 char
        for line in file:
            for char in line:
                # arr.append(IntegerConversionFunction(char))
                if char == ' ':
                    arr.append(int(0))
                elif char == '+':
                    arr.append(int(1))
                elif char == '#':
                    arr.append(int(2))
            training_digits.append(arr)
            # init the arr for the new line/row
            arr = []
            row_count = row_count + 1
            if row_count == n:
                row_count = 0
                result.append(training_digits)
                # init the arr for the new line/row
                training_digits = []
        result = np.array(result)
    print(result.shape)
    return result
