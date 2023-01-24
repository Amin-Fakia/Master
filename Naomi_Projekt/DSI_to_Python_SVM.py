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
from sklearn.preprocessing import MinMaxScaler
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
mne.set_log_level("ERROR")
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
		for i in range (len(data1)): # evt. lamda funktion + zusammenfassen 
			val = data1[i]-data3[i]
			C3_minus_Cz_data.append(val)
		for i in range (len(data2)):
			val = data2[i]-data3[i]
			C4_minus_Cz_data.append(val)
		for i in range (len(C4_minus_Cz_data)):
			val = C4_minus_Cz_data[i]-C3_minus_Cz_data[i]
			C4_minus_Cz_C3_data.append(val)
		return C4_minus_Cz_C3_data

	def minmaxscaler(self, data):	
		filename="minmaxscaler.pkl"
		scaler3 = pickle.load(open(filename, "rb"))
		return scaler3.transform(data)
		

	def probability_decision_right_left(self, result):	
		if result[0, 0]>=0.53:
			return("links")   
		elif result[0, 1] >=0.53:
			return("rechts")
		else:
			return("keine Bewegung")

		
	def real_time(self):
		data_thread = threading.Thread(target=self.parse_data)
		data_thread.start()
		
		sample_freq = 300
		refresh_rate = 1/sample_freq
		channels = ["P3","C3","F3","Fz","F4","C4","P4","Cz","CM",
					"A1","Fp1","Fp2","T3","T5","O1","O2","X3","X2",
					"F7","F8","X1","A2","T6","T4","TRG"] 
		#sub_channels = ["O1","O2"]
		
		# change this to your model name
		filename ='./Naomi_Projekt/finalized_model_prob_svc_7261%_ohne_ica.sav'
		# imgs = [mpimg.imread("Bild1.png"),mpimg.imread("Bild2.png")]
		loaded_model = pickle.load(open(filename, "rb"))
		C3_index = channels.index("C3")
		C4_index = channels.index("C4")
		Cz_index = channels.index("Cz")

		fig,axs = plt.subplots(1,2)
		
		labels = ["left","right"]

		while True:
			
			data_raw = self.signal_log[:,-self.packet_size:] # O1, O2 => 14:16
			time_log = self.time_log[0,-self.packet_size:] #
			plt.clf()
			
			plt.axis("off")
			try:
				
				plt.ylim((-50,50))
				
				# Variante 1 mit array
				# Filter the data
				clean_data = mne.filter.filter_data(data_raw,sfreq=sample_freq,l_freq=3,h_freq=30) # 3 Hz Hochpass, 30 Hz Tiefpassfilter
				
			
				# Get the data from only the interesting channels
				C3_data =clean_data[C3_index]
				C4_data = clean_data[C4_index]
				Cz_data = clean_data[Cz_index]

				# Do some calculations
				X = self.do_calculation(C3_data,C4_data,Cz_data)

				
				# scaled_X = self.minmaxscaler(X)
				
				norm_X = X / np.std(X)
				
				
				result = loaded_model.predict_proba([norm_X])
				erg = self.probability_decision_right_left(result)

				






				# # Variante 2 mit edf File
				# # create edf file
				# info = mne.create_info(ch_names = channels, sfreq=300, ch_types='eeg', verbose=None)
				# edf_data = mne.io.RawArray(data_raw, info, first_samp=0, copy='auto', verbose=None)

				# # filter data
				# edf_data.load_data().filter(3, 30)
				

				# # Get the data from only the interesting channels				
				# C3_data1 = edf_data.get_data(picks='C3')
				# C4_data1 = edf_data.get_data(picks='C4')
				# Cz_data1 = edf_data.get_data(picks='Cz')
				# print("C3_data1-------------")
				# print(C3_data1)

				# # ica - set montage and apply saved ica to data



				# # apply baseline correction
				# C3_data1_resc = mne.baseline.rescale(C3_data1, time_log, baseline=(None, None), mode='mean', copy=True)
				# C4_data1_resc = mne.baseline.rescale(C4_data1, time_log, baseline=(None, None), mode='mean', copy=True)
				# Cz_data1_resc = mne.baseline.rescale(Cz_data1, time_log, baseline=(None, None), mode='mean', copy=True)
				# print("C3_data1_resc-------------")
				# print(C3_data1_resc)



				# # Do some calculations
				# X = self.do_calculation(C3_data1_resc,C4_data1_resc,Cz_data1_resc)
				# print("X-------------")
				# print(X)
				# X = np.array(X)
				# X = np.concatenate(X)
				
				# X_train_2d = X.reshape(1, -1) 
				# X_train_2d = X_train_2d / np.std(X_train_2d)

				# # scaled_X = self.minmaxscaler(X)
				

				# norm_X = X_train_2d / np.linalg.norm(X_train_2d) #X
				# result = loaded_model.predict_proba([norm_X]) 
				# erg = self.probability_decision_right_left(result)
				

			
				
				
				#plt.text(.25,.5,f"{result}")
				
				plt.text(.25,.5, f"{erg} mit Wahrscheinlichkeit für \n rechts von {result[0, 1]:.2f} % \n links {result[0, 0]:.2f}")
				
				# plt.text(.25,.5,f"{result[0]} -> {labels[int(result[0])]}")


			except Exception as e: plt.text(0.1,0.5,e) # pass #return e # pass ## 
			
			plt.pause(refresh_rate)
			
		
			
		plt.show()



		

		


		
if __name__ == "__main__":

	
	tcp = TCPParser('localhost',9067,1) # provide the duration you would like to have in seconds, if you want 100 datapoints, then provide 1/3 => 1/3 * 300 (sample frequency) = 100 data points
	tcp.real_time()
		
		
		