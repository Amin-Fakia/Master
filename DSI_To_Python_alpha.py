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
from scipy.signal import butter, lfilter,welch
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
from scipy.integrate import simpson
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
		N = sample_freq 
		X1 = channels.index("X1")
		X2 = channels.index("X2")
		X3 = channels.index("X3")
		xf = rfftfreq(N-1, 1 / sample_freq)
		n_oneside = N//2
		fig,ax = plt.subplots()
		while True:
			data_raw = self.signal_log[:len(channels)-1,-self.packet_size:]
			time_log = self.time_log[0,-self.packet_size:]
			
			# get only sub-array channels - X1, X2, X3

		
			
			try:
				plt.clf()
				power_values = []
				pv_relative=[]
				for idx,data in enumerate(data_raw):
					clean_data = mne.filter.filter_data(data,sfreq=sample_freq,l_freq=1,h_freq=50,method='iir')
					clean_data = np.delete(clean_data, X1)
					clean_data = np.delete(clean_data, X2)
					clean_data = np.delete(clean_data, X3)

					freqs, psd = welch(clean_data,sample_freq)
					freq_res = freqs[1] - freqs[0]
					
					idx_Theta_Alpha = np.logical_and(freqs>=4,freqs<=12)
					Theta_Alpha_power = simpson(psd[idx_Theta_Alpha], dx=freq_res)
					power_values.append(Theta_Alpha_power)
					total_power = simpson(psd, dx=freq_res)
					Theta_Alpha_rel_power = (Theta_Alpha_power / total_power)*100
					# print('Relative Theta_Alpha power: %.3f' % Theta_Alpha_rel_power
					
					total_power = simpson(psd, dx=freq_res)
					Theta_Alpha_rel_power = (Theta_Alpha_power / total_power)*100
					pv_relative.append(Theta_Alpha_rel_power)
				print(f'Absolute Theta_Alpha power: {np.average(power_values):.3f} uV^2')
				print(f'Relative Theta_Alpha power: {np.average(pv_relative):.3f} %')
				

			except Exception as e: print(e) #pass
			
			
			plt.pause(5)
			
		
	
		


		
if __name__ == "__main__":

	# The script will automatically run the example_plot() method if not called from another script.
	tcp = TCPParser('localhost',9067,5) # provide the duration you would like to have in seconds, if you want 100 datapoints, then provide 1/3 => 1/3 * 300 (sample frequency) = 100 data points
	
	
	tcp.real_time()
	
	
	#tcp.real_time()

	
