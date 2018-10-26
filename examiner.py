import numpy as np
from solver import pos2grid


def examiner(table):
    for idx in range(9):
        for value in set(table[idx, :]):
            if np.count_nonzero(table[idx, :] == value) > 1:
                return False
        for value in set(table[:, idx]):
            if np.count_nonzero(table[:, idx] == value) > 1:
                return False
    for row_idx, col_idx in pos2grid.keys():
        grid_values = list()
        for row in range(row_idx * 3, (row_idx + 1) * 3):
            for col in range(col_idx * 3, (col_idx + 1) * 3):
                grid_values.append(table[row, col])
        for value in grid_values:
            if grid_values.count(value) > 1:
                return False
    return True


if __name__ == '__main__':
    test_table = np.array([[9, 6, 3, 1, 7, 4, 2, 5, 8],
                           [1, 7, 8, 3, 2, 5, 6, 4, 9],
                           [2, 5, 4, 6, 8, 9, 7, 3, 1],
                           [8, 2, 1, 4, 3, 7, 5, 9, 6],
                           [4, 9, 6, 8, 5, 2, 3, 1, 7],
                           [7, 3, 5, 9, 6, 1, 8, 2, 4],
                           [5, 8, 9, 7, 1, 3, 4, 6, 2],
                           [3, 1, 7, 2, 4, 6, 9, 8, 5],
                           [6, 4, 2, 5, 9, 8, 1, 7, 3]])

    test_table2 = np.array([[9, 8, 2, 1, 7, 3, 5, 4, 6],
                            [6, 7, 1, 5, 9, 4, 8, 3, 2],
                            [3, 5, 4, 2, 6, 8, 7, 9, 1],
                            [2, 9, 3, 6, 8, 5, 1, 7, 4],
                            [5, 4, 6, 7, 1, 9, 3, 2, 8],
                            [7, 1, 8, 3, 4, 2, 9, 6, 5],
                            [8, 2, 9, 4, 5, 7, 6, 1, 3],
                            [4, 6, 7, 8, 3, 1, 2, 5, 9],
                            [1, 3, 5, 9, 2, 6, 4, 8, 7]])

    test_table3 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [9, 1, 2, 3, 4, 5, 6, 7, 8],
                            [8, 9, 1, 2, 3, 4, 5, 6, 7],
                            [7, 8, 9, 1, 2, 3, 4, 5, 6],
                            [6, 7, 8, 9, 1, 2, 3, 4, 5],
                            [5, 6, 7, 8, 9, 1, 2, 3, 4],
                            [4, 5, 6, 7, 8, 9, 1, 2, 3],
                            [3, 4, 5, 6, 7, 8, 9, 1, 2],
                            [2, 3, 4, 5, 6, 7, 8, 9, 1]])

    print(examiner(test_table))
