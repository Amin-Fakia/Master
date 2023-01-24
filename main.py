import numpy as np
from vispy.app import use_app, Timer
from vispy.scene import SceneCanvas,visuals
from PyQt5 import QtWidgets, QtCore
from math import pi,sin

CANVAS_SIZE = (1200, 600)  # (width, height)
NUM_LINE_POINTS = 200
LINE_COLORS = ["red", "black", "blue"]


class MyMainWindow(QtWidgets.QMainWindow):
    def __init__(self,canvas_wrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._controls = Controls()
        main_layout.addWidget(self._controls)
        self._canvas_wrapper = canvas_wrapper
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self._connect_controls()

    def _connect_controls(self):
        self._controls.line_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_line_color)



class Controls(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
       
        
        self.line_color_label = QtWidgets.QLabel("Line color:")
        layout.addWidget(self.line_color_label)
        self.line_color_chooser = QtWidgets.QComboBox()
        self.line_color_chooser.addItems(LINE_COLORS)
        layout.addWidget(self.line_color_chooser)

        layout.addStretch(1)
        self.setLayout(layout)

class CanvasWrapper:
    def __init__(self,size,bgcolor="black",view_bgcolor="white"):
        self.canvas = SceneCanvas(size=CANVAS_SIZE,bgcolor=bgcolor)
        self.grid = self.canvas.central_widget.add_grid()
        self.lines =[]
        line_data = _init_line_positions(NUM_LINE_POINTS)
        for i in range(size):
            self.view = self.grid.add_view(i, 0, bgcolor=view_bgcolor)
            self.view.camera = "panzoom"
            self.view.camera.set_range(x=(0, NUM_LINE_POINTS), y=(0, 1))
            line = visuals.Line(line_data,parent=self.view.scene,color=LINE_COLORS[0],width=2)
            self.lines.append(line)
    def set_line_color(self,color):
        #print("changing line color to " + str(color))
        for line in self.lines:
            line.set_data(color=color)

 
    def update_data(self,new_data_dict):
        #print("updating data ...")
        for line in self.lines:
            line.set_data(new_data_dict["line"])

      
def _init_line_positions(num_points, dtype=np.float32):
    pos = np.empty((num_points, 2), dtype=np.float32)
    pos[:, 0] = np.arange(num_points)
    pos[:, 1] = np.ones((num_points,),dtype=dtype)
    return pos       
        

class DataSource(QtCore.QObject):
    new_data = QtCore.pyqtSignal(dict)

    def __init__(self, num_iterations=1000,parent=None):
        super().__init__(parent)
        self._count = 0
        self._num_iters = num_iterations
        self._line_data = _init_line_positions(NUM_LINE_POINTS)
    def run_data_creation(self,timer_event):
        if self._count >= self._num_iters:
            return
        line_data = self._update_line_data(self._count)
        self._count +=1
        data_dict= {"line":line_data}
        self.new_data.emit(data_dict)
    
    def _update_line_data(self,count):
        self._line_data[:,1] = np.roll(self._line_data[:,1],-1)
        self._line_data[-1,1] = abs(sin((count/self._num_iters)*32*pi))
        return self._line_data.copy()

        
# def _generate_random_line_positions(num_points, dtype=np.float32):
#     rng = np.random.default_rng()
#     pos = np.empty((num_points, 2), dtype=np.float32)
#     pos[:, 0] = np.arange(num_points)
#     pos[:, 1] = rng.random((num_points,), dtype=dtype)
#     return pos


if __name__ == "__main__":
    app = use_app("pyqt5")
    app.create()
    
    data_source = DataSource()
    canvas_wrapper = CanvasWrapper(12)

    win = MyMainWindow(canvas_wrapper)

    data_source.new_data.connect(canvas_wrapper.update_data)
    timer = Timer(f"{1/120}",connect=data_source.run_data_creation,start=True)


    win.show()

    app.run()



