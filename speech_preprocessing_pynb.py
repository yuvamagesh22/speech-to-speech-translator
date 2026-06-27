
from IPython.lib.display import Audio
#saudio files loding phase:
import librosa;
import librosa.display;
import matplotlib.pyplot as plt;
import numpy as np

audio_path = "/content/2428-83705-0035.flac";
audio , sr = librosa.load(audio_path, sr = None );
duration = librosa.get_duration(y=audio, sr=sr)
speech,_ = librosa.effects.trim(audio); #Keep only the spoken parts.
D = librosa.stft(audio); #Convert speech from the time domain to the frequency domain.
mel = librosa.feature.melspectrogram(y = audio, sr= sr); #Represents sound in a way that is closer to human hearing.
mfcc = librosa.feature.mfcc(y = audio, sr = sr); # Extracts features commonly used in speech recognition.

 #remove silence:
plt.figure(figsize=(12 ,3));
librosa.display.waveshow(speech , sr=sr);
plt.title("Remove silence")
plt.show()

#visulisation 1. waveform (timedomain signal)

plt.figure(figsize=(12, 4));
librosa.display.waveshow(audio , sr = sr);
plt.title("Waveform (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

#2. STFT:(frequency domain signal)
DB = librosa.amplitude_to_db(np.abs(D), ref = np.max);
plt.figure(figsize=(12, 5));
librosa.display.specshow(DB, sr= sr, x_axis="time", cmap= "magma");
plt.colorbar(format="%+2.0f dB")
plt.title("STFT Spectrogram")
plt.show()
#3. Mel Spectrogram (human hearing scale)
mp = librosa.power_to_db( mel , ref=np.max);
librosa.display.specshow(mp, sr=sr, x_axis="time", y_axis="mel", cmap="viridis");
plt.colorbar(format="%+2.0f db");
plt.title("spectrogram");
plt.show()
#4. mfcc spectrogram:
plt.figure(figsize=(12 , 3))
librosa.display.specshow(mfcc, sr = sr, x_axis="time", cmap = "coolwarm");
plt.colorbar();
plt.title("Mfcc");
plt.show()

print(duration)
print("samplingrate", sr);
print("no of samples",len(audio));
print("audio length ",len(speech));
print("frequency over time :", D);
print("spectrogram", mel);
print("Result",mfcc);


