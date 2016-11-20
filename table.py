from PySide import QtGui, QtCore


class Table(QtGui.QWidget):
    padding = 10

    def __init__(self):
        super(Table, self).__init__()
        self.setWindowTitle("Sudoku Table")
        # self.setMinimumSize(300, 300)
        self.setFixedSize(300, 300)
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
                painter.drawRect(QtCore.QRect(i * main_rect.width() / 9 + self.padding,
                                              j * main_rect.height() / 9 + self.padding,
                                              main_rect.width() / 9,
                                              main_rect.height() / 9))
        painter.end()
        return


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    table = Table()
    table.show()
    sys.exit(app.exec_())
