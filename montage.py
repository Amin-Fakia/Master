import mne
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import time
import vedo
dsi_channels= "P3,C3,F3,Fz,F4,C4,P4,Cz,CM,A1,Fp1,Fp2,T3,T5,O1,O2,X3,X2,F7,F8,X1,A2,T6,T4,TRG"
channels= ["P3","C3","F3","Fz","F4","C4","P4","Cz","CM","A1","Fp1","Fp2","T3","T5","O1","O2","X3","X2","F7","F8","X1","A2","T6","T4","TRG"]


def get_1020eeg_positions(channels):
    montage_1020 = mne.channels.make_standard_montage('standard_1020')
    positions = montage_1020.get_positions()['ch_pos']
    chnls_pos = {}
    for idx,pos in enumerate(positions.values()):
        if list(positions.keys())[idx] in channels:
            chnls_pos[list(positions.keys())[idx]] = pos

    #return ordered_dict
    return dict(sorted(chnls_pos.items(), key=lambda pair: channels.index(pair[0])))

ch_pos = get_1020eeg_positions(dsi_channels)


pg.mkQApp()
view = gl.GLViewWidget()

scaled_values = [30*i for i in list(ch_pos.values())]

plot = gl.GLScatterPlotItem()
plot.setData(pos=scaled_values,size=15)
view.addItem(plot)

for idx,s in enumerate(scaled_values):
    text = gl.GLTextItem()
    text.setData(pos=s,text=list(ch_pos.keys())[idx])
    view.addItem(text)
    
    


view.show()
for i in range(100):
    random = np.random.rand(len(scaled_values),3)
    plot.setData(color=random)
    time.sleep(1/300)
    QtGui.QGuiApplication.processEvents()




if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QGuiApplication.exec_()


 
