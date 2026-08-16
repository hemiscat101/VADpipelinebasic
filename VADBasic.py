import librosa
librosa.load
import numpy as np
from pydub import AudioSegment
import os
import torch
import librosa
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

audio = AudioSegment.from_file(r"D:\VADBASIC\260217_0025_1-2.wav")

HF_TOKEN = "." #your hf token here

speaker_files = {
    "nikhil": r"D:\VADbasic\fourspeaker3pm nikhilLapel.wav",
    "paddy": r"D:\VADbasic\fourspeaker3pm paddyLapel.wav",
    "siddMTech": r"D:\VADbasic\fourspeaker3pm siddMTechLapel.wav",
    "siddPhd": r"D:\VADbasic\fourspeaker3pm siddPhdLapel.wav"   
    }

session_id = "fourspeaker"
output_rttm = rf"D:\VADBASIC\{session_id}.rttm"

model = Model.from_pretrained("pyannote/segmentation-3.0", token=HF_TOKEN)

vad_pipeline = VoiceActivityDetection(segmentation=model)
vad_pipeline.instantiate({
    #"onset": 0.5,
    #"offset": 0.5,
    "min_duration_on": 0.25,
    "min_duration_off": 0.1
})

def get_segments(path):
    wav, sr = librosa.load(path, sr=16000, mono=True)
    waveform = torch.from_numpy(wav).unsqueeze(0) 
    audio_dict = {"waveform": waveform, "sample_rate": sr}
    output = vad_pipeline(audio_dict)
    return [(seg.start, seg.end) for seg in output.get_timeline()]

lines = []
for spk, path in speaker_files.items():
    segs = get_segments(path)
    print(f"{spk}: {len(segs)} segments")
    for start, end in segs:
        dur = end - start
        lines.append((start, f"SPEAKER {session_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"))

lines.sort(key=lambda x: x[0])

with open(output_rttm, "w") as f:
    f.write("\n".join(l for _, l in lines) + "\n")

print(f"\n{len(lines)} total segments written to {output_rttm}")