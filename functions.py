
from timeit import repeat
import numpy as np
import mne
import scipy.interpolate as si
from vedo import *
from scipy.interpolate import Rbf,RBFInterpolator
from scipy.signal import savgol_filter
from vedo import show, interactive
import matplotlib.pyplot as plt
import time
from PyQt5 import QtWidgets
from matplotlib.widgets import SpanSelector
import keyboard
import matplotlib.animation as animation

from matplotlib.backend_bases import MouseButton
import scipy
from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter, writers

mne.set_log_level(0)



def get_intrpContour(data,x,y,xi,yi):
    zs = []
    for d in range(len(data[0])):
        zs.append(scipy.interpolate.griddata((x, y), [i[d] for i in data], (xi[None,:], yi[:,None]), method='cubic'))
    return zs
def get_data_from_raw_edf(raw):
    data = raw.get_data()[0:len(raw.get_data())-1]
    f_data = []
    
    for i in range(0,len(data)):
        if i == 13:
            f_data.append(data[13])
        elif i == 14:
            f_data.append(data[14])
        else:
            f_data.append([0]*len(data[i]))
    return f_data 

def get_times(raw):
    df = raw.to_data_frame()
    return df.iloc[:,0]
def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
def plot_data_from(raw,channels):
    fig,ax = plt.subplots(len(raw))
    c = 0
    for d in raw:

        ax[c].plot(range(len(d)),d, c='blue',linewidth=0.5)
        ax[c].set_ylabel(f"{channels[c]}", rotation=0)
       
        clean_ax(ax[c])
        c+=1
    plt.show()
def plot_data_from_edf(raw,channels):
    fig,ax = plt.subplots(len(raw.get_data()))
    times = [t/1000 for t in get_times(raw)]
    c = 0
    for d in get_data_from_raw_edf(raw):
        ax[c].plot(times,d, c='blue',linewidth=0.5)
        ax[c].set_ylabel(f"{channels[c]}", rotation=0)
       
        clean_ax(ax[c])
        c+=1
    plt.show()
def plot_data(data):
    fig,ax = plt.subplots()
    ax.plot(range(0,len(data)), data)
    plt.show()
def get_text(t1,t2):
    return Text2D(f'{t1/1000} - {t2/1000} in s',s=2,c='r')   
def get_average(data):
    return np.average(data,axis=0)
def animate_data_span(raw,mesh,pts):
    fig,ax = plt.subplots()
    times = [t/1000 for t in get_times(raw)]
    for d in get_data_from_raw_edf(raw):
        line, = ax.plot(times,d,linewidth=0.5,c='blue')
        
    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(times, (xmin, xmax))
        indmax = min(len(times) - 1, indmax)
        
        region_x = times[indmin:indmax]
        plt.close() 
        animate(mesh,pts,raw,times.index(min(region_x)),times.index(max(region_x)),0.01)


    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='tab:red'))
    try:
        win = fig.canvas.manager.window
    except AttributeError:
        win = fig.canvas.window()
    toolbar = win.findChild(QtWidgets.QToolBar)
    toolbar.setVisible(False)
    clean_ax(ax)
    plt.show()
def get_sensors_from_montage(ch_names,monatge_type="standard_1020",curc=.56):
    montage = mne.channels.make_standard_montage(f"{monatge_type}",curc).get_positions()['ch_pos']
    
    i = 0
    c_ = []
    for c in ch_names:
        
        if c == "EEG Fp1-Pz" or c == "EEG Fp2-Pz":
            c_.append(c[4:7])
        else:
            c_.append(c[4:6])
        c_
    
    print(c_)
    ch_pos= {}
    print(montage.keys())
    for j in c_:
        if j in list(montage.keys()):
            ch_pos[j] = montage[j]
    # for k in montage:
        
    #     for j in c_:
    #         print(j)
    #         if(j == k):
    #             print(k)
    #             ch_pos[j] = montage[k]
    return ch_pos
def get_mesh(s):
    if isinstance(s,str):
        mesh = Mesh(s)
        mesh.clean().normalize()
        mesh.rotateX(110) # 90
        mesh.rotateZ(180)
        mesh.origin(0,-0.015,-0.04) # mesh.origin(0,-0.015,-0.05), latest: mesh.origin(0,-0.01,-0.04)
        #mesh.origin(-0.01,-0.03,-0.04)

        mesh.scale(0.09)#  mesh.scale(0.09)
        return mesh
def get_sensor_2DLocations(l,exl=[""]):
    pts = []
    for i, k in l.items():
        if i not in exl:
            pts.append([k[0],k[1]])
    return pts
def get_sensor_3DLocations(l,exl=[""]):
    pts = []
    for i, k in l.items():
        if i not in exl:
            pts.append([k[0],k[1],k[2]])
    return pts
def smoothFilter(data,winsize=71,po=3):
    data_smooth = []
    for d in data:
        data_smooth.append(savgol_filter(d,winsize,po))
    return data_smooth
def getRGB(actor, alpha=True, on='points'):
    """
        Get RGB(A) colors from a vedo Actor
        :param actor: Vedo Mesh Object
        :param bool alpha: to include/exclude alpha
        :param string on: points or cells
    """
    lut = actor.mapper().GetLookupTable()
    poly = actor.polydata(transformed=False)
    if 'point' in on:
        vscalars = poly.GetPointData().GetScalars()
    else:
        vscalars = poly.GetCellData().GetScalars()
    cols =lut.MapScalars(vscalars, 0,0)
    arr = utils.vtk2numpy(cols)
    if not alpha:
        arr = arr[:, :3]
    return arr
def get_power_values(data,sampling_rate,win_size=3,step=.1,tmin=None,tmax =None):
    """
        Get the power values based on : multi-channel data, sampling frequency, window size, step
        optional cut data from tmin, tmax
    """

    min_win = 0
    max_win = win_size
    data_array = []
    while(max_win < (len(data[0])/sampling_rate)):
        pos_x1 = int((min_win)*sampling_rate)
        pos_x2 = int((max_win)*sampling_rate)
        
        sums = []
        
        for d in data:
            ft = np.abs(np.fft.rfft(d[pos_x1:pos_x2]))
            ps = np.square(ft)
            
            #sums.append(sum(ps))
            sums.append(sum(ps))
        
        data_array.append(sums)
        min_win +=step
        max_win +=step
    return np.transpose(data_array)
def get_ERP_values(data,sampling_rate,win_size,step,itr):
    min_win = 0
    max_win = win_size
    data_array = []
    while(max_win < itr):
        pos_x1 = int((min_win)*sampling_rate)
        pos_x2 = int((max_win)*sampling_rate)
        sums = []
        for d in data:
            sums.append(sum(d[pos_x1:pos_x2]))
        data_array.append(sums)
        min_win +=step
        max_win +=step
    return np.transpose(data_array)   
def dist(p1, p2):
     
    x0 = p1[0] - p2[0]
    y0 = p1[1] - p2[1]
    return x0 * x0 + y0 * y0
def maxDist(p):
 
    n = len(p)
    maxm = 0
 
    # Iterate over all possible pairs
    for i in range(n):
        for j in range(i + 1, n):
             
            # Update maxm
            maxm = max(maxm, dist(p[i], p[j]))
 
    # Return actual distance
    return sqrt(maxm)
# find the shortest distance between a sensor point and mesh points
def findMinD(x,pts,mesh):
    dist = []
    for p in mesh.points():
        dist.append(np.linalg.norm(pts[x]-p))
    return dist.index(min(dist))
# find the corresponding point coordinates on the mesh 
def findVert(pts,mesh):
    vrt =[]
    for i in range(0,len(pts)):
        vrt.append(findMinD(i,pts,mesh))
    return [mesh.points()[i] for i in vrt]

def Linear_Interpolation(mesh,pts,data):

    xi, yi, zi = np.split(mesh.points(), 3, axis=1) 
    lir = si.LinearNDInterpolator(pts,data)
    return [[i] for i in np.squeeze(lir(xi, yi, zi))]
def enhanced_RBF(data):
    pass

def RBF_Interpolation(mesh,pts,data,function="gaussian"):
    x, y, z = np.split(np.array(pts), 3, axis=1)
    itr = Rbf(x,y,z,data,function=function)
    xi, yi, zi = np.split(mesh.points(), 3, axis=1)
    return itr(xi,yi,zi)

def plot_edf(raw):
    raw.plot(duration=30)
    plt.show()
def plot_tfa(raw):
    raw.plot_psd()
    plt.show()
   
def animate(mesh,pts,raw,t1,t2,f=1,text=''):
    # Rbf or Linear

    data = get_data_from_raw_edf(raw)
    times = get_times(raw)
    
    #print(len(times))
    if t1 < 0 and t1 < t2:
        print("please insert a valid starting time-point")
    if t2 > len(data[0]) and t2 > t1 :
        print("please insert a valid ending time-point")
    vmin = min([min(i[t1:t2]) for i in data])
    #[t1:t2]
    vmax = max([max(i[t1:t2]) for i in data])
    
    text = get_text(times[t1],times[t2])
    
    points= Points(pts,r=9,alpha=0.7,c='w')
    plot = show(interactive=False,bg='k')
    datas= []
    
    for i in range(t1,t2):
        text2 = Text2D(f'\n \n {times[i]/1000} ')
        intpr = RBF_Interpolation(mesh,pts,[j[i] for j in data])
        datas = [l[i] for l in data]
        points.cmap('jet', datas, vmin=vmin, vmax=vmax)
        mesh.cmap('jet', intpr, vmin=vmin, vmax=vmax)
        plot.show(mesh,points,text,text2) # ,text2
        plot.remove(text)
        plot.remove(text2)
        time.sleep(f)
    plot.close()
    quit()

def plot_window_with_ax(data,sampling_rate,win_size,step,event_time,tmin = None,tmax = None):

    #times = [i*(1/sampling_rate) for i in range(len(data))]
    if tmin == None or tmax == None:
        times = np.arange(0,len(data)/300,1/sampling_rate)
        tmin = times[0]
        tmax = times[-1]
    else:
        times = np.arange(tmin,tmax,1/sampling_rate)
        data = data[tmin*300:tmax*300]
    
    # times = np.arange(0,len(data)/300,1/sampling_rate)
    
    
    
    #times = times[tmin*300:tmax*300]
    #data = data[tmin*300:tmax*300]
    #test_data = data[tmin*300:tmax*300]
    ft = np.abs(np.fft.rfft(data))
    ps = np.square(ft)
    frequency = np.linspace(0, sampling_rate/2, len(ps))
    fig, ax = plt.subplots(3)
    fig.set_size_inches((15, 10))
    # ax[0].axvspan(0,times[int(len(times)/3)], color='red',alpha=0.2)
    # ax[0].axvspan(times[int(len(times)/3)],times[int(len(times)/1.1)], color='green',alpha=0.2)
    # ax[0].axvspan(times[int(len(times)/1.1)],times[int(len(times))-1], color='red',alpha=0.2)
    
    min_win = 0
    max_win = win_size
    ln, = ax[1].plot(frequency, ft)
    def init():
        ax[1].set_ylim((0,10e-8))
    
    data_y_smooth = []
    
    
    def update(frame):

        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        nonlocal max_win
        nonlocal min_win

        ax[0].plot(times,data,label="EEG O1",color='k',lw=0.5)
        ax[0].set_xlabel("time in s")
        ax[0].set_ylabel("Amplitude in V")
        ax[0].axvline(event_time, ls='--',color='blue',lw=2, label='Eyes-Closed')
        ax[0].axvspan(min_win+tmin,max_win+tmin, color='red',alpha=0.5)
        ax[0].legend()
        pos_x1 = int((min_win)*sampling_rate)

        pos_x2 = int((max_win)*sampling_rate)
        
        ft = np.abs(np.fft.rfft(data[pos_x1:pos_x2]))
        ps = np.square(ft)
        frequency = np.linspace(0, sampling_rate/2, len(ps))
        ax[1].set_ylim((0,1e-5))
        ax[1].set_ylabel("Intensity (arb. u.)")
        ax[1].set_xlabel("Frequency in Hz")
        ax[1].plot(frequency,ps, color='red',label=(f'Sum: {sum(ps)*(10**5):.2f} e-5'))
        ax[1].legend()

        data_y_smooth.append(sum(ps))


        ax[2].set_ylabel("Power Value (sum)")
        ax[2].set_xlabel("Time Frame")
        ax[2].set_ylim((1e-6,20e-6))
        ax[2].set_xlim((0,750))
        ax[2].plot(range(len(data_y_smooth)),data_y_smooth, color='red',alpha=0.5,label=(f'{sum(ps)*(10**5):.2f} e-5'))
        ax[2].legend()

        min_win +=step
        max_win +=step


    
    ax[0].plot(times,data)
    ax[0].set_xlabel("time in s")
    ax[0].set_ylabel("Amplitude in V")
    ax[0].legend(["Time Window"])
    print("size",tmax-win_size)
    ani = FuncAnimation(fig, update, frames=np.arange(tmin,tmax-win_size,step),init_func=init,save_count=0,interval=50,blit = False,repeat=False)
    #writegif = PillowWriter(fps=60)
    Writer = writers['ffmpeg']
    writer = Writer(fps=60,metadata={'artist':'Me'},codec="h264", bitrate=1000000)
    ani.save('Final.mp4',writer)
    # writer = Writer(fps=60, bitrate=3000)
    #ani.save("FFTwindows.gif",writer=writegif,dpi=200)
    # axnext = plt.axes([0.88, 0.05, 0.1, 0.075])
    # bnext = Button(axnext, 'Next')
    #plt.show()
    # bnext.on_clicked(nxt)
def animate_vline(data,event_time):
    fig, ax = plt.subplots()
    ax.plot(range(len(data)), data)
    fig.set_size_inches((11, 2))
    vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
    ax.axvline(event_time, ls='--', color='b', lw=1,label="Eyes Closed Event")
    ax.legend()
    def update(frame):
        vl.set_xdata(frame)
        return vl,
        


    ani = FuncAnimation(fig, update, frames=len(data),interval=1/60,blit=True,repeat=False)
    Writer = writers['ffmpeg']
    writer = Writer(fps=60,metadata={'artist':'Me'},codec="h264", bitrate=1000000)
    #ani.save('vlineAnimation.mp4',writer)
    plt.show()
def plot_window_with_vedo(data,sampling_rate,win_size,step,mesh,sensor_pts,event_time,tmin = None,tmax = None):
    #from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    #times = [i*(1/sampling_rate) for i in range(len(data))]
    if tmin == None or tmax == None:
        times = np.arange(0,len(data)/300,1/sampling_rate)
        tmin = times[0]
        tmax = times[-1]
    else:
        times = np.arange(tmin,tmax,1/sampling_rate)
        data = data[tmin*300:tmax*300]
    
    # times = np.arange(0,len(data)/300,1/sampling_rate)
    # vtkWidget = QVTKRenderWindowInteractor()
    # vp = Plotter(qtWidget=vtkWidget)

    
    
    #times = times[tmin*300:tmax*300]
    #data = data[tmin*300:tmax*300]
    #test_data = data[tmin*300:tmax*300]
    ft = np.abs(np.fft.rfft(data))
    ps = np.square(ft)
    frequency = np.linspace(0, sampling_rate/2, len(ps))
    fig, ax = plt.subplots(3)
    fig.set_size_inches((15, 12))
    # ax[0].axvspan(0,times[int(len(times)/3)], color='red',alpha=0.2)
    # ax[0].axvspan(times[int(len(times)/3)],times[int(len(times)/1.1)], color='green',alpha=0.2)
    # ax[0].axvspan(times[int(len(times)/1.1)],times[int(len(times))-1], color='red',alpha=0.2)
    
    min_win = 0
    max_win = win_size
    ln, = ax[1].plot(frequency, ft)
    
    def init():
        ax[1].set_ylim((0,10e-8))

    def update(frame):

        ax[0].cla()
        ax[1].cla()
        nonlocal max_win
        nonlocal min_win

        ax[0].plot(times,data,label="EEG O1",color='k',lw=0.5)
        ax[0].set_xlabel("time in s")
        ax[0].set_ylabel("Amplitude in V")
        ax[0].axvline(event_time, ls='--',color='blue',lw=2, label='Eyes-Closed')
        ax[0].axvspan(min_win+tmin,max_win+tmin, color='red',alpha=0.5)
        ax[0].legend()
        pos_x1 = int((min_win)*sampling_rate)

        pos_x2 = int((max_win)*sampling_rate)
        
        ft = np.abs(np.fft.rfft(data[pos_x1:pos_x2]))
        ps = np.square(ft)
        frequency = np.linspace(0, sampling_rate/2, len(ps))
        ax[1].set_ylim((0,1e-5))
        ax[1].set_ylabel("Intensity (arb. u.)")
        ax[1].set_xlabel("Frequency in Hz")
        ax[1].plot(frequency,ps, color='red',label=(f'Sum: {sum(ps)*(10**5):.2f} e-5'))
        ax[1].legend()
        min_win +=step
        max_win +=step

    ax[0].plot(times,data)
    ax[0].set_xlabel("time in s")
    ax[0].set_ylabel("Amplitude in V")
    ax[0].legend(["Time Window"])

        
    