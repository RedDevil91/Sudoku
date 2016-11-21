import numpy as np

pos2grid = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (1, 0): 3,
    (1, 1): 4,
    (1, 2): 5,
    (2, 0): 6,
    (2, 1): 7,
    (2, 2): 8,
}


class EmptyBox(object):
    def __init__(self, row, col, options=None):
        self.row = row
        self.col = col
        self.options = options
        self.fill_value = None
        return


class Grid(object):
    def __init__(self, idx):
        self.empty_boxes = list()
        self.fill_values = list()
        self.index = idx
        return

    def addEmptyBox(self, box):
        self.empty_boxes.append(box)
        return

    def addFillValue(self, value):
        self.fill_values.append(value)
        return

    def findUniqeBoxes(self):
        uniqe_values = self.findUniqeValues()
        for value in uniqe_values:
            for box in self.empty_boxes:
                if value in box.options:
                    box.fill_value = value
                    break
        return

    def findUniqeValues(self):
        unique_values = list()
        for box in self.empty_boxes:
            unique_values += box.options if box.fill_value is None else []
        unique_values = [val for val in unique_values if (unique_values.count(val) == 1 and
                                                          val not in self.fill_values)]
        return unique_values


class RowOrColumn(object):
    def __init__(self):
        self.empty_boxes = list()
        self.fill_values = list()
        return

    def addEmptyBox(self, box):
        self.empty_boxes.append(box)
        return

    def addFillValue(self, value):
        self.fill_values.append(value)
        return


class Solver(object):
    def __init__(self, table):
        self.table = table
        self.empty_cells = self.findEmptyCells()
        return

    def findEmptyCells(self):
        empty_cells = list()
        for row in xrange(9):
            for col in xrange(9):
                if self.table[row, col] == 0:
                    empty_cells.append(EmptyBox(row, col))
        return empty_cells

    def solve(self):
        self.checkRowOptions()
        self.checkColOptions()
        self.checkGridOptions()
        self.updateEmptyBoxes()
        if len(self.empty_cells) != 0:
            self.solve()
            return
        return

    def updateOptions(self):
        for box in list(self.empty_cells):
            row_numbers = self.checkRow(box.row)
            col_numbers = self.checkCol(box.col)
            box_numbers = self.checkBox(box.row, box.col)
            not_numbers = list(row_numbers.union(col_numbers, box_numbers))
            options = [opt for opt in range(1, 10) if opt not in not_numbers]
            box.options = options
            if len(box.options) == 1:
                box.fill_value = box.options[0]
        return

    def checkRow(self, row):
        excl_numbers = set(self.table[row, :])
        excl_numbers.remove(0)
        return excl_numbers

    def checkCol(self, col):
        excl_numbers = set(self.table[:, col])
        excl_numbers.remove(0)
        return excl_numbers

    def checkBox(self, row, col):
        excl_numbers = set()
        box_row, box_col = self.getGridPos(row, col)
        for row in xrange(box_row * 3, (box_row + 1) * 3):
            for col in xrange(box_col * 3, (box_col + 1) * 3):
                excl_numbers.add(self.table[row, col])
        excl_numbers.remove(0)
        return excl_numbers

    def checkRowOptions(self):
        self.updateOptions()
        rows = [RowOrColumn() for r in range(9)]
        for box in list(self.empty_cells):
            if not box.fill_value:
                rows[box.row].addEmptyBox(box)
            else:
                rows[box.row].addFillValue(box.fill_value)
        for row in rows:
            self.findUniqueBoxes(row)
        return

    def checkColOptions(self):
        self.updateOptions()
        cols = [RowOrColumn() for c in range(9)]
        for box in list(self.empty_cells):
            if not box.fill_value:
                cols[box.col].addEmptyBox(box)
            else:
                cols[box.col].addFillValue(box.fill_value)
        for col in cols:
            self.findUniqueBoxes(col)
        return

    def checkGridOptions(self):
        self.updateOptions()
        for grid in self.createSubGrids():
            grid.findUniqeBoxes()
        return

    def createSubGrids(self):
        grids = [Grid(idx) for idx in range(9)]
        for box in list(self.empty_cells):
            r, c = self.getGridPos(box.row, box.col)
            if not box.fill_value:
                grids[pos2grid[(r, c)]].addEmptyBox(box)
            else:
                grids[pos2grid[(r, c)]].addFillValue(box.fill_value)
        return grids

    def findUniqueBoxes(self, box_list):
        unique_values = self.findUniqeValues(box_list)
        for value in unique_values:
            for box in box_list.empty_boxes:
                if value in box.options:
                    box.fill_value = value
                    break
        return

    def findUniqeValues(self, box_list):
        unique_values = list()
        for box in box_list.empty_boxes:
            unique_values += box.options if box.fill_value is None else []
        unique_values = [val for val in unique_values if (unique_values.count(val) == 1 and
                                                          val not in box_list.fill_values)]
        return unique_values

    def updateEmptyBoxes(self):
        for box in list(self.empty_cells):
            if box.fill_value:
                self.table[box.row][box.col] = box.fill_value
                self.empty_cells.remove(box)
        return

    def getGridPos(self, row, col):
        return row / 3, col / 3

    def setValue(self, box):
        self.table[box.row, box.col] = box.fill_value
        if box in self.empty_cells:
            self.empty_cells.remove(box)
        else:
            print box
        return

if __name__ == '__main__':
    import sys
    puzzle = np.array([[6, 0, 0, 0, 2, 0, 0, 0, 9],
                       [0, 1, 0, 3, 0, 7, 0, 5, 0],
                       [0, 0, 3, 0, 0, 0, 1, 0, 0],
                       [0, 9, 0, 0, 0, 0, 0, 2, 0],
                       [2, 0, 0, 8, 7, 5, 0, 0, 3],
                       [0, 0, 5, 0, 1, 0, 4, 0, 0],
                       [0, 7, 0, 0, 8, 0, 0, 9, 0],
                       [0, 0, 1, 0, 4, 0, 8, 0, 0],
                       [0, 0, 0, 2, 5, 9, 0, 0, 0]])

    puzzle2 = np.array([[0, 8, 0, 1, 0, 0, 0, 0, 0],
                        [6, 0, 0, 0, 0, 4, 8, 0, 0],
                        [3, 5, 4, 2, 6, 0, 7, 9, 1],
                        [2, 9, 3, 0, 8, 0, 0, 0, 4],
                        [0, 0, 0, 7, 0, 9, 0, 0, 0],
                        [7, 0, 0, 0, 4, 0, 9, 6, 5],
                        [8, 2, 9, 0, 5, 7, 6, 1, 3],
                        [0, 0, 7, 8, 0, 0, 0, 0, 9],
                        [0, 0, 0, 0, 0, 6, 0, 8, 0]])

    puzzle3 = np.array([[0, 6, 0, 1, 0, 4, 0, 5, 0],
                        [0, 0, 8, 3, 0, 5, 6, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0, 0, 1],
                        [8, 0, 0, 4, 0, 7, 0, 0, 6],
                        [0, 0, 6, 0, 0, 0, 3, 0, 0],
                        [7, 0, 0, 9, 0, 1, 0, 0, 4],
                        [5, 0, 0, 0, 0, 0, 0, 0, 2],
                        [0, 0, 7, 2, 0, 6, 9, 0, 0],
                        [0, 4, 0, 5, 0, 8, 0, 7, 0]])

    solver = Solver(puzzle)
    try:
        solver.solve()
        print solver.table
    except RuntimeError:
        print "Can't find solution!"
    sys.exit(0)
