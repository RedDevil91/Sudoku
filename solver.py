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
        self.boxes = list()
        self.index = idx
        return

    def addBox(self, box):
        self.boxes.append(box)
        return

    def findUniqeBox(self):
        boxes = list()
        uniqe_values = self.findUniqeValue()
        for value in uniqe_values:
            for box in self.boxes:
                if value in box.options:
                    box.fill_value = value
                    boxes.append(box)
                    break
        return boxes

    def findUniqeValue(self):
        opts = list()
        for box in self.boxes:
            opts += box.options
        for num in set(opts):
            if opts.count(num) != 1:
                opts = [opt for opt in opts if opt != num]
        return opts


class Solver(object):
    def __init__(self, table):
        self.table = table
        self.empty_cells = self.findEmptyCells()
        # print self.empty_cells
        return

    def findEmptyCells(self):
        empty_cells = list()
        for row in xrange(9):
            for col in xrange(9):
                if self.table[row, col] == 0:
                    empty_cells.append(EmptyBox(row, col))
        return empty_cells

    def solve(self):
        # for box in list(self.empty_cells):
        #     self.getOptions(box)
        self.checkRowOptions()
        # self.checkColOptions()
        # self.checkGridOptions()
        if len(self.empty_cells) != 0:
            self.solve()
        return self.table

    def getOptions(self, box):
        row_numbers = self.checkRow(box.row)
        col_numbers = self.checkCol(box.col)
        box_numbers = self.checkBox(box.row, box.col)
        not_numbers = list(row_numbers.union(col_numbers, box_numbers))
        options = [opt for opt in range(1, 10) if opt not in not_numbers]
        box.options = options
        if len(box.options) == 1:
            box.fill_value = box.options[0]
            self.setValue(box)
            return True
        return False

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
        rows = [[] for r in range(9)]
        for box in list(self.empty_cells):
            if not self.getOptions(box):
                rows[box.row].append(box)
        val_boxes = list()
        for row in rows:
            boxes = self.findUniqueBox(row)
            val_boxes += boxes
            # for box in boxes:
            #     self.setValue(box)
        for box in val_boxes:
            self.setValue(box)
        return

    def checkColOptions(self):
        cols = [[] for c in range(9)]
        for box in list(self.empty_cells):
            if not self.getOptions(box):
                cols[box.col].append(box)
        val_boxes = list()
        for col in cols:
            boxes = self.findUniqueBox(col)
            val_boxes += boxes
            # for box in boxes:
            #     self.setValue(box)
        for box in val_boxes:
            self.setValue(box)
        return

    def checkGridOptions(self):
        val_boxes = list()
        for grid in self.createGrids():
                boxes = grid.findUniqeBox()
                val_boxes += boxes
                # for box in boxes:
                #     self.setValue(box)
        for box in val_boxes:
            self.setValue(box)
        return

    def createGrids(self):
        grids = [Grid(idx) for idx in range(9)]
        for box in list(self.empty_cells):
            if not self.getOptions(box):
                r, c = self.getGridPos(box.row, box.col)
                grids[pos2grid[(r, c)]].addBox(box)
        return grids

    def getGridPos(self, row, col):
        return row / 3, col / 3

    def findUniqueBox(self, box_list):
        boxes = list()
        uniqe_values = self.findUniqeValue(box_list)
        for value in uniqe_values:
            for box in box_list:
                if value in box.options:
                    box.fill_value = value
                    boxes.append(box)
                    break
        return boxes

    def findUniqeValue(self, box_list):
        opts = list()
        for box in box_list:
            opts += box.options
        for num in set(opts):
            if opts.count(num) != 1:
                opts = [opt for opt in opts if opt != num]
        return opts

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

    solver = Solver(puzzle3)
    try:
        print solver.solve()
    except RuntimeError:
        print "Can't find solution!"
    sys.exit(0)
