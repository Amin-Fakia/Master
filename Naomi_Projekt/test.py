import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler

def do_calculation(data1,data2,data3):
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


# channels = ["P3","C3","F3","Fz","F4","C4","P4","Cz","CM",
# 					"A1","Fp1","Fp2","T3","T5","O1","O2","X3","X2",
# 					"F7","F8","X1","A2","T6","T4","TRG"]


# C3_index = channels.index("C3")
# C4_index = channels.index("C4")
# Cz_index = channels.index("Cz")


channels = ['EEG Fp1-Vref', 'EEG Fp2-Vref', 'EEG Fz-Vref',
 'EEG F3-Vref', 'EEG F4-Vref', 'EEG F7-Vref', 'EEG F8-Vref', 'EEG Cz-Vref', 
 'EEG C3-Vref', 'EEG C4-Vref', 'EEG T3-Vref', 'EEG T4-Vref', 'EEG T5-Vref', 'EEG T6-Vref', 
 'EEG Pz-Vref', 'EEG P3-Vref', 'EEG P4-Vref', 
'EEG O1-Vref', 'EEG O2-Vref', 'EEG A1-Vref', 'EEG A2-Vref', 'Trigger']


C3_index = channels.index("EEG C3-Vref")
C4_index = channels.index("EEG C4-Vref")
Cz_index = channels.index("EEG Cz-Vref")

raw = mne.io.read_raw_edf("./Naomi_Projekt/test_data/rechts_1811_1_M_raw.edf")

events = mne.find_events(raw)
epochs = mne.Epochs(raw,events,tmin=-0.1,tmax=0.9)

print(epochs)
raw_data = epochs.get_data()


clean_data = mne.filter.filter_data(raw_data,sfreq=300,l_freq=3,h_freq=30)
scaler = MinMaxScaler()

scaled = scaler.fit_transform(np.reshape(clean_data[0][0],(-1,1)))



# print(len(clean_data[0]))
C3_data =[c[C3_index] for c in clean_data]
C4_data = [c[C4_index] for c in clean_data]
Cz_data = [c[Cz_index] for c in clean_data]





filename ='./Naomi_Projekt/KI_Model/finalized_model_svc1_76%.sav'

loaded_model = pickle.load(open(filename, "rb"))

counterLinks = 0
counterRechts = 0
# try:


for i in range(len(C3_data)):
    X = do_calculation(C3_data[i],C4_data[i],Cz_data[i])
    norm_X = X / np.std(X)
    # scaled = scaler.fit_transform(np.reshape(X,(-1,1)))
    # scaled = scaled.reshape(1,-1)
    erg = loaded_model.predict([norm_X])
    print(erg)

    if(erg[0] < 1):
        counterLinks+=1
    else:
        counterRechts+=1
        
# except: pass

print("Links ", counterLinks)
print("Rechts ", counterRechts)
print("Insgesamt ", counterLinks + counterRechts)  

