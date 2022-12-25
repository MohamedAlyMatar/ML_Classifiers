# importing modules
from samples import *
import numpy as np


def read_lines2(filepath, n):
    result = []
    with open(filepath, 'r', encoding='utf-8') as file:
        # for 28 sample
        for sample in range(n):
            # for 28 rows
            # init the arr for the new line/row
            arr = []
            for row in range(28):
                # for each 28 char
                for ch in range(28):
                    # read by character
                    char = file.read(1)
                    # check if it is a char or not
                    if not char:
                        break
                    arr.append(IntegerConversionFunction(char))
            result.append(arr)
    result = np.array(result)
    print(result.shape)
    return result


# 5000x784
def read2(filename, n):
    with open(filename, 'r') as file:
        result = []
        lines = file.readlines()
        for sample in range(n):
            data = []
            for row in range(28):
                for col in range(28):
                    data.append(convertToInteger(lines[row][col]))
            result.append(data)
    return np.array(result)


def read(filename, n):
    with open(filename, 'r') as file:
        result = []
        lines = file.readlines()
        for sample in range(n):
            datumn = []
            for row in range(28):
                data = []
                for col in range(28):
                    data.append(convertToInteger(lines[row][col]))
                datumn.append(data)
            result.append(datumn)
    return np.array(result)


def read_lines(filepath):
    result = []
    training_digits = []

    with open(filepath, 'r', encoding='utf-8') as file:
        # for 28 image
        for letter in range(28):
            # for 28 rows
            for row in range(28):
                # init the arr for the new line/row
                arr = []
                # for each 28 char
                for ch in range(28):
                    # read by character
                    char = file.read(1)
                    # check if it is a char or not
                    if not char:
                        break
                    arr.append(IntegerConversionFunction(char))
                training_digits.extend(arr)
            result.append(training_digits)
    result = np.array(result)
    print(result.shape)
    return result


def read_labels(filepath):
    arr = []
    with open(filepath, "r", encoding='utf-8') as file:
        for line in file:
            line = line.rstrip("\n")
            arr.append(line)
    barr = [int(i) for i in arr]
    arr = np.array(barr)
    return arr


# print(read_lines2('data/digitdata/trainingimages', 5000))
# read_labels("data/digitdata/traininglabels")
