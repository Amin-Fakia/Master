import sys
from PyQt5 import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
# You may need to uncomment these lines on some systems:
#import vtk.qt
#vtk.qt.QVTKRWIBase = "QGLWidget"

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Cone
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.vl = Qt.QVBoxLayout()

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
       

        self.vl.addWidget(sc)
        

        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        # create renderer and add the actors
        self.vp = Plotter(qt_widget=self.vtkWidget)

        self.vp += Cone().rotateX(20).alpha(0.8)

        self.vp.show(interactive=0)

        # set-up the rest of the Qt window
        button = Qt.QPushButton("My Button makes the cone red")

        #button.clicked.connect(self.onClick)
        self.vl.addWidget(button)
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.show() # <--- show the Qt Window





    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        self.vtkWidget.close()

    @Qt.pyqtSlot()
    def onClick(self):
        self.vp.actors[0].color('red').rotateZ(40)
        self.vp.interactor.Render()

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    app.exec_()