# DSI_to_Python v.1.0 BETA
# The following script can be used to receive DSI-Streamer Data Packets through DSI-Streamer's TCP/IP Protocol.
# It contains an example parser for converting packet bytearrays to their corresponding formats described in the TCP/IP Socket Protocol Documentation (https://wearablesensing.com/downloads/TCPIP%20Support_20190924.zip).
# The script involves opening a server socket on DSI-Streamer and connecting a client socket on Python.

# As of v.1.0, the script outputs EEG data and timestamps to the command window. In addition, the script is able to plot the data in realtime.
# Keep in mind, however, that the plot functionality is only meant as a demonstration and therefore does not adhere to any current standards.
# The plot function plots the signals on one graph, unlabeled.
# To verify correct plotting, one can introduce an artifact in the data and observe its effects on the plots.

# The sample code is not certified to any specific standard. It is not intended for clinical use.
# The sample code and software that makes use of it, should not be used for diagnostic or other clinical purposes.  
# The sample code is intended for research use and is provided on an "AS IS"  basis.  
# WEARABLE SENSING, INCLUDING ITS SUBSIDIARIES, DISCLAIMS ANY AND ALL WARRANTIES
# EXPRESSED, STATUTORY OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OR THIRD PARTY RIGHTS.
#
# Copyright (c) 2014-2020 Wearable Sensing LLC
# P3,C3,F3,Fz,F4,C4,P4,Cz,CM,A1,Fp1,Fp2,T3,T5,O1,O2,X3,X2,F7,F8,X1,A2,T6,T4,TRG
import time
import socket, struct, time
import numpy as np
import matplotlib.pyplot as plt
import threading
from scipy.signal import butter, lfilter
import matplotlib.image as mpimg
import mne
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from vispy import plot as vp
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from vispy import app, scene
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget

mne.set_log_level("ERROR")
plt.style.use('ggplot')
class TCPParser: # The script contains one main class which handles DSI-Streamer data packet parsing.

	def __init__(self, host, port,duration=1):
		"""
			host: string -> localhost
			port: int -> DSI Client Inport
			duration: int -> in seconds
		"""
		self.host = host
		self.port = port
		self.done = False
		self.data_log = b''
		self.latest_packets = []
		self.latest_packet_headers = []
		self.latest_packet_data = np.zeros((1,1))
		self.signal_log = np.zeros((1,20))
		self.time_log = np.zeros((1,20))
		self.montage = []
		self.data = []
		self.fsample = 0
		self.fmains = 0

		self.packet_size = int(duration * 301)

		self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		self.sock.connect((self.host,self.port))
	
	def parse_data(self):
		
		# parse_data() receives DSI-Streamer TCP/IP packets and updates the signal_log and time_log attributes
		# which capture EEG data and time data, respectively, from the last 100 EEG data packets (by default) into a numpy array.
		while not self.done:
			data = self.sock.recv(921600)
			self.data_log += data
			if self.data_log.find(b'@ABCD',0,len(self.data_log)) != -1:										# The script looks for the '@ABCD' header start sequence to find packets.
				for index,packet in enumerate(self.data_log.split(b'@ABCD')[1:]):							# The script then splits the inbound transmission by the header start sequence to collect the individual packets.
					self.latest_packets.append(b'@ABCD' + packet)
				for packet in self.latest_packets:
					self.latest_packet_headers.append(struct.unpack('>BHI',packet[5:12]))
				self.data_log = b''


				for index, packet_header in enumerate(self.latest_packet_headers):		
					# For each packet in the transmission, the script will append the signal data and timestamps to their respective logs.
					if packet_header[0] == 1:
						if np.shape(self.signal_log)[0] == 1:												# The signal_log must be initialized based on the headset and number of available channels.
							self.signal_log = np.zeros((int(len(self.latest_packets[index][23:])/4),20))
							self.time_log = np.zeros((1,20))
							self.latest_packet_data = np.zeros((int(len(self.latest_packets[index][23:])/4),1))

						self.latest_packet_data = np.reshape(struct.unpack('>%df'%(len(self.latest_packets[index][23:])/4),self.latest_packets[index][23:]),(len(self.latest_packet_data),1))
						self.latest_packet_data_timestamp = np.reshape(struct.unpack('>f',self.latest_packets[index][12:16]),(1,1))

						self.signal_log = np.append(self.signal_log,self.latest_packet_data,1)
						self.time_log = np.append(self.time_log,self.latest_packet_data_timestamp,1)
						self.signal_log = self.signal_log[:,-self.packet_size:]
						self.time_log = self.time_log[:,-self.packet_size:]
					## Non-data packet handling
					if packet_header[0] == 5:
						(event_code, event_node) = struct.unpack('>II',self.latest_packets[index][12:20])
						if len(self.latest_packets[index])>24:
							message_length = struct.unpack('>I',self.latest_packets[index][20:24])[0]
						#print("Event code = " + str(event_code) + "  Node = " + str(event_node))
						if event_code == 9:
							montage = self.latest_packets[index][24:24+message_length].decode()
							montage = montage.strip()
							print("Montage = " + montage)
							self.montage = montage.split(',')
						if event_code == 10:
							frequencies = self.latest_packets[index][24:24+message_length].decode()
							#print("Mains,Sample = "+ frequencies)
							mains,sample = frequencies.split(',')
							self.fsample = float(sample)
							self.fmains = float(mains)
			self.latest_packets = []
			self.latest_packet_headers = []

	def do_calculation(self,data1,data2,data3):
		C3_minus_Cz_data = []
		C4_minus_Cz_data = []
		C4_minus_Cz_C3_data = []
		for i in range (len(data1)):
			val = data1[i]-data3[i]
			C3_minus_Cz_data.append(val)
		for i in range (len(data2)):
			val = data2[i]-data3[i]
			C4_minus_Cz_data.append(val)
		for i in range (len(C4_minus_Cz_data)):
			val = C4_minus_Cz_data[i]-C3_minus_Cz_data[i]
			C4_minus_Cz_C3_data.append(val)
		return C4_minus_Cz_C3_data


	def real_time(self):
		data_thread = threading.Thread(target=self.parse_data)
		data_thread.start()
		
		sample_freq = 300
		refresh_rate = 1/sample_freq
		channels = ["P3","C3","F3","Fz","F4","C4","P4","Cz","CM",
					"A1","Fp1","Fp2","T3","T5","O1","O2","X3","X2",
					"F7","F8","X1","A2","T6","T4","TRG"]
		sub_channels = ["O1","O2"]

		band_range = {'Delta':[0.05,3],'Theta':[3.05,8],'Alpha':[8.05,12],'Beta':[12.05,30],'Gamma':[30.05,49.75]}
		C3_index = channels.index("C3")


		N = sample_freq 
		xf = rfftfreq(N-1, 1 / sample_freq)
		n_oneside = N//2

		
		pg_layout = pg.GraphicsLayoutWidget(show=True)
		#pg_layout.setBackground('w')
		
		# plot = pg_layout.addPlot(row=0,col=0)
		# plot_signal = pg_layout.addPlot(row=1,col=0)
		
		
		# all signals
		plot_signals =[]
		freq_plots = []
		
		chnls_num = 2
		
		for i in range(chnls_num):
			plot = pg_layout.addPlot(row=i,col=0)
			plot.hideAxis('left')
			plot.hideAxis('bottom')
			plot.addLegend()
			plot_signals.append(plot)
		for i in range(chnls_num):
			plot = pg_layout.addPlot(row=i,col=1)
			# plot.hideAxis('left')
			# plot.hideAxis('bottom')
			plot.addLegend()
			freq_plots.append(plot)        
		#plot.showGrid(x = True, y = True)
		#plot_signal.showGrid(x = True, y = True)
		
		
		# plot.setYRange(0, 3000, padding=0)
		# plot_signal.setYRange(-20, 20, padding=0)
		# plot.addLegend()
		# plot_signal.addLegend()
		
	
		
	
		print(list(band_range.values()))
		
		while True:
			data_raw = self.signal_log[14:16,-self.packet_size:] # O1, O2
			#data_raw = self.signal_log[:len(channels)-1,-self.packet_size:] # O1, O2
			time_log = self.time_log[0,-self.packet_size:] #
			try:
				

				# plot.clear()
				# plot_signal.clear()
				
				

				
		
				# Filter the data
				clean_data_IIR = mne.filter.filter_data(data_raw,sfreq=sample_freq,l_freq=0.5,h_freq=50,method='iir')
				#clean_data_IIR = data_raw
				for idx, p in enumerate(plot_signals):
					
					p.clear()
					
					
					curve = pg.PlotCurveItem(y=clean_data_IIR[idx],pen=pg.mkPen(color=idx),name=f"{sub_channels[idx]}",)
					
					p.addItem(curve)
				eeg_band_fft = dict()
				

				for idx, pl in enumerate(freq_plots):
					
					pl.clear()
					yf = np.abs(np.fft.fft(clean_data_IIR[idx]))
					yf = yf[:n_oneside]

					for band in band_range:  
						freq_ix = np.where((xf >= band_range[band][0]) & 
										(xf <= band_range[band][1]))[0]
						eeg_band_fft[band] = np.mean(yf[freq_ix])
					

					#,pen=pg.mkPen(color=idx),name=f"FFT: {sub_channels[idx]}",fillLevel=-0.3, brush=(10*idx,10*idx,200,70),
					#curve = pg.BarGraphItem(x=xf,height=yf,width=2,bin=list(band_range.values()),pen=pg.mkPen(color=idx),name=f"FFT: {sub_channels[idx]}",fillLevel=-0.3, brush=(10*idx,10*idx,200,70))
					curve = pg.BarGraphItem(x=range(len(list(eeg_band_fft.values()))),height=list(eeg_band_fft.values()),
								width=0.9,
								pen=pg.mkPen(color=idx),
								fillLevel=-0.3, brush=(10*idx,10*idx,200,70))
					
					
					pl.addItem(curve)
					
					ax = pl.getAxis('bottom')
					ticks=[]
					xval = list(range(0,len(list(eeg_band_fft.keys()))))
					for i, item in enumerate(list(eeg_band_fft.keys())):
						ticks.append( (xval[i], item) )
					ticks = [ticks]
					
					ax.setTicks(ticks)
				
				
				#clean_data_IIR =data_raw
				#clean_data_FIR = mne.filter.filter_data(data_raw,sfreq=sample_freq,l_freq=8,h_freq=12,method='fir')
				
				# yf1 = np.abs(np.fft.fft(clean_data_IIR[0]))
				# yf2 = np.abs(np.fft.fft(clean_data_IIR[1]))
				# yf1 = yf1[:n_oneside]
				# yf2 = yf2[:n_oneside]

				# #yf = (yf1+yf2)/2
				# curve = pg.PlotCurveItem(xf,yf1,pen=pg.mkPen(color=(50, 50, 200)),fillLevel=-0.3, brush=(50,50,200,50), name=f"FFT {sub_channels[0]} ")
				# curve2 = pg.PlotCurveItem(xf,yf2,pen=pg.mkPen(color=(255, 191, 0)),fillLevel=-0.3, brush=(255, 191, 0,50), name=f"FFT {sub_channels[1]} ")
				# plot.addItem(curve)
				# plot.addItem(curve2)


				# signal_1 = pg.PlotCurveItem(time_log,clean_data_IIR[0],pen=pg.mkPen(color=(50, 50, 200)), name=f"{sub_channels[0]} ")
				# signal_2 = pg.PlotCurveItem(time_log,clean_data_IIR[1],pen=pg.mkPen(color=(255, 191, 0)), name=f"{sub_channels[0]} ")
				# plot_signal.addItem(signal_1)
				# plot_signal.addItem(signal_2)

				
			except Exception as e: print(e) #pass
			#plt.pause(refresh_rate)
			QtGui.QGuiApplication.processEvents()
			time.sleep(refresh_rate)
			

		
	
		


		
if __name__ == "__main__":

	# The script will automatically run the example_plot() method if not called from another script.
	tcp = TCPParser('localhost',9067,4) # provide the duration you would like to have in seconds, if you want 100 datapoints, then provide 1/3 => 1/3 * 300 (sample frequency) = 100 data points
	
	
	tcp.real_time()
	
	
	#tcp.real_time()

	
