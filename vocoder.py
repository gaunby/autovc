# Mel-Spectrograms to waveform
import torch
import pickle
from synthesis import build_model
from synthesis import wavegen

import soundfile as sf

spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth", map_location=torch.device(device))
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    sf.write(name+'.wav', waveform,16000)

