from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget
import numpy as np


class Table(QWidget):
    table_size = 300
    padding = 10

    def __init__(self, init_values):
        super(Table, self).__init__()
        self.setWindowTitle("Sudoku Table")
        self.setGeometry(100, 100, self.table_size, self.table_size)
        self.setFixedSize(self.table_size, self.table_size)

        self.values = init_values
        return

    def paintEvent(self, event):
        pen = QtGui.QPen(QtCore.Qt.black, 4)
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setPen(pen)

        main_rect = QtCore.QRect(self.padding, self.padding,
                                 self.width() - 2 * self.padding,
                                 self.height() - 2 * self.padding)

        [self.drawRect(painter, main_rect.width() // 3, main_rect.height() // 3, i, j)
            for i in range(3) for j in range(3)]

        pen.setWidth(1)
        painter.setPen(pen)

        [self.drawRectWithValue(painter, main_rect.width() // 9, main_rect.height() // 9, i, j)
            for i in range(9) for j in range(9)]

        painter.end()
        return

    def drawRect(self, painter, width, height, row, col):
        rect = QtCore.QRect(row * width + self.padding, col * height + self.padding, width, height)
        painter.drawRect(rect)
        return rect

    def drawRectWithValue(self, painter, width, height, row, col):
        rect = self.drawRect(painter, width, height, row, col)
        if self.values[row, col]:
            painter.drawText(rect, QtCore.Qt.AlignCenter, str(self.values[col, row]))
        return


if __name__ == '__main__':
    import sys
    puzzle2 = np.array([[0, 8, 0, 1, 0, 0, 0, 0, 0],
                        [6, 0, 0, 0, 0, 4, 8, 0, 0],
                        [3, 5, 4, 2, 6, 0, 7, 9, 1],
                        [2, 9, 3, 0, 8, 0, 0, 0, 4],
                        [0, 0, 0, 7, 0, 9, 0, 0, 0],
                        [7, 0, 0, 0, 4, 0, 9, 6, 5],
                        [8, 2, 9, 0, 5, 7, 6, 1, 3],
                        [0, 0, 7, 8, 0, 0, 0, 0, 9],
                        [0, 0, 0, 0, 0, 6, 0, 8, 0]])

    app = QApplication(sys.argv)
    table = Table(puzzle2)
    table.show()
    sys.exit(app.exec_())
