import os
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wv
from pydub import AudioSegment, effects


url = sys.argv[1]
subprocess.call(f"youtube-dl -x '{url}' --audio-format m4a -o 'AUDIOs'", shell=True)


def rm(fn):
    subprocess.call(f"rm -rf {fn}", shell=True)


def conv_audio(fn, ff):
    subprocess.call(
        f"ffmpeg -i {fn} -loglevel quiet {os.path.splitext(fn)[0]}.{ff}", shell=True
    )
    rm(fn)


def c_itm(lst, itm):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - itm))]


[rm(x) for x in ["AUDIO.m4a", "hello.png"]]
conv_audio("AUDIO.m4a", "wav")
rate, audData = wv.read("AUDIO.wav")
subprocess.call("rm -rf out; mkdir out", shell=True)


channel1 = audData[:, 0]
channel2 = audData[:, 1]

n = len(channel1)
fourier = np.fft.fft(channel1)[0 : round(n / 2)] / float(n)


freqArray = (np.arange(0, (n / 2), 1.0) * (rate * 1.0 / n)) / 1000
db_array = 10 * np.log10(fourier)

Pxx, freqs, timebins, im = plt.specgram(
    channel2, Fs=rate, NFFT=1024, noverlap=0, cmap=plt.get_cmap("autumn_r")
)


MHZ10 = Pxx[233, :]
pltspec.figure(figsize=(8, 6))
pltspec.plot(timebins, MHZ10, color="#ff7f00")


lstoutbins = {
    "60": [],
    "150": [],
    "400": [],
    "1000": [],
    "2400": [],
    "15000": [],
}
mapvals = {
    "60": 2,
    "150": 25,
    "400": 52,
    "1000": 38,
    "2400": 90,
    "15000": 2,
}
lstcombs = []
finalaudio = 0
for x in range(len(MHZ10)):
    try:
        og_number = f"{float(MHZ10[x]):f}"
        closest = c_itm([int(x) for x in list(lstoutbins.keys())], float(og_number))
        lstoutbins[str(closest)].append(timebins[x])
        if x != len(MHZ10) - 1:
            segment = [x, x + 1]
        else:
            segment = [x, len(MHZ10)]

        lstcombs.append(
            [
                int(round(timebins[segment[0]] * 1000)),
                int(round(timebins[segment[1]] * 1000)),
            ]
        )
        song = AudioSegment.from_wav("AUDIO.wav")
        extract = song[
            int(round(timebins[segment[0]] * 1000)) : int(
                round(timebins[segment[1]] * 1000)
            )
        ]
        extract.export(f"out/{x}-extract.wav", format="wav")
        extracted = AudioSegment.from_wav(f"out/{x}-extract.wav")
        sys.stdout.write(f"{round(MHZ10[x])} ({x}/{len(MHZ10)})\r")
        sys.stdout.flush()
        extracted = extracted - (extracted.dBFS * (mapvals[str(closest)] / 100))
        subprocess.call(f"rm -rf out/{x}-extract.wav", shell=True)
        if finalaudio == 0:
            finalaudio = extracted
        else:
            finalaudio = finalaudio + extracted
    except IndexError:
        continue

normalizedsound = effects.normalize(finalaudio)
normalizedsound.export("out/outfile.wav", format="wav")
