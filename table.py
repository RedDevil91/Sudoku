from PySide import QtGui, QtCore
import numpy as np


class Table(QtGui.QWidget):
    padding = 10

    def __init__(self, init_values):
        super(Table, self).__init__()
        self.setWindowTitle("Sudoku Table")
        self.setGeometry(100, 0, 300, 300)
        self.setFixedSize(300, 300)

        self.values = init_values
        return

    def paintEvent(self, event):
        height = self.height()
        width = self.width()

        pen = QtGui.QPen(QtCore.Qt.black, 4)
        painter = QtGui.QPainter()
        painter.begin(self)

        painter.setPen(pen)
        main_rect = QtCore.QRect(self.padding, self.padding,
                                 width - 2 * self.padding,
                                 height - 2 * self.padding)
        for i in xrange(3):
            for j in xrange(3):
                painter.drawRect(QtCore.QRect(i * main_rect.width() / 3 + self.padding,
                                              j * main_rect.height() / 3 + self.padding,
                                              main_rect.width() / 3,
                                              main_rect.height() / 3))
        pen.setWidth(1)
        painter.setPen(pen)
        for i in xrange(9):
            for j in xrange(9):
                rect = QtCore.QRect(i * main_rect.width() / 9 + self.padding,
                                    j * main_rect.height() / 9 + self.padding,
                                    main_rect.width() / 9,
                                    main_rect.height() / 9)
                painter.drawRect(rect)
                if self.values[j, i] != 0:
                    painter.drawText(rect, QtCore.Qt.AlignCenter, str(self.values[j, i]))
        painter.end()
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

    app = QtGui.QApplication(sys.argv)
    table = Table(puzzle2)
    table.show()
    sys.exit(app.exec_())
