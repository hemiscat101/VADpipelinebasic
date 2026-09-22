import numpy as np
import librosa
from collections import defaultdict


def smooth_energy(energy_db, window=5):
    if window <= 1:
        return energy_db
    kernel = np.ones(window) / window
    return np.convolve(energy_db, kernel, mode="same")


def load_channels(speaker_files, sr=16000):
    audio = {}
    for spk, path in speaker_files.items():
        wav, _ = librosa.load(path, sr=sr, mono=True)
        audio[spk] = wav

    min_len = min(len(w) for w in audio.values())
    for spk in audio:
        audio[spk] = audio[spk][:min_len]

    return audio, sr


def energy(wav, frame_len, hop_len):
    frames = librosa.util.frame(wav, frame_length=frame_len, hop_length=hop_len)
    energy = np.sqrt(np.mean(frames ** 2, axis=0))
    return energy


def per_channel_threshold(energy_db, floor_percentile=10, margin_db=12):
    noise_floor = np.percentile(energy_db, floor_percentile)
    return noise_floor + margin_db


def resolve_same_speaker_overlaps(segments, bridge_s=0.0):
    by_spk = defaultdict(list)
    for spk, s, e in segments:
        by_spk[spk].append((s, e))

    resolved = []
    for spk, ranges in by_spk.items():
        ranges.sort()
        merged = []
        for s, e in ranges:
            if merged and s <= merged[-1][1] + bridge_s:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        for s, e in merged:
            resolved.append((spk, s, e))

    resolved.sort(key=lambda x: x[1])
    return resolved


def diarization(speaker_files, session_id, output_rttm,
                 sr=16000, frame_ms=25, hop_ms=10,
                 floor_percentile=10, silence_margin_db=12,
                 dominance_margin_db=3,
                 smooth_window=5, pad_ms=200, bridge_gap_ms=150):

    audio, sr = load_channels(speaker_files, sr=sr)
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    speakers = list(audio.keys())

    energies = np.stack([energy(audio[spk], frame_len, hop_len) for spk in speakers])
    energies_db = 20 * np.log10(np.maximum(energies, 1e-10))  # avoid log(0)
    energies_db = np.stack([smooth_energy(energies_db[i], smooth_window)
                             for i in range(energies_db.shape[0])])

    thresholds = np.array([
        per_channel_threshold(energies_db[i], floor_percentile, silence_margin_db)
        for i in range(len(speakers))
    ])
    print("Per-speaker noise floor + margin thresholds (dB):",
          {spk: round(float(t), 1) for spk, t in zip(speakers, thresholds)})

    n_frames = energies_db.shape[1]
    frame_labels = [None] * n_frames

    for i in range(n_frames):
        frame_vals = energies_db[:, i]
        max_idx = np.argmax(frame_vals)
        max_val = frame_vals[max_idx]

        if max_val < thresholds[max_idx]:
            continue

        sorted_vals = np.sort(frame_vals)[::-1]
        if len(sorted_vals) > 1 and (sorted_vals[0] - sorted_vals[1]) < dominance_margin_db:
            continue

        frame_labels[i] = speakers[max_idx]

    segments = []
    current_spk = None
    seg_start = None

    for i, label in enumerate(frame_labels):
        t = i * hop_len / sr
        if label != current_spk:
            if current_spk is not None:
                segments.append((current_spk, seg_start, t))
            current_spk = label
            seg_start = t
    if current_spk is not None:
        segments.append((current_spk, seg_start, n_frames * hop_len / sr))

    
    pad_s = pad_ms / 1000.0
    file_end = n_frames * hop_len / sr
    padded = [(spk, max(0.0, s - pad_s), min(file_end, e + pad_s)) for spk, s, e in segments]

    bridge_s = bridge_gap_ms / 1000.0
    segments = resolve_same_speaker_overlaps(padded, bridge_s=bridge_s)

    lines = []
    for spk, start, end in segments:
        dur = end - start
        lines.append((start, f"SPEAKER {session_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"))

    lines.sort(key=lambda x: x[0])
    with open(output_rttm, "w") as f:
        f.write("\n".join(l for _, l in lines) + "\n")

    print(f"{len(segments)} segments written to {output_rttm}")
    return segments


if __name__ == "__main__":
    speaker_files = {
        "Lapel1": r"/DATA/VAD/04-lapel1-260831_1329.wav",
        "Lapel2": r"/DATA/VAD/05-lapel2-260831_1329.wav",
        "Lapel3": r"/DATA/VAD/06-lapel3-260831_1329.wav",
        "Lapel4": r"/DATA/VAD/07-lapel4-260831_1329.wav"
    }

    diarization(
        speaker_files=speaker_files,
        session_id="pilot",
        output_rttm=r"D:\VADBASIC\pilot.rttm",
        floor_percentile=5,
        silence_margin_db=14,
        dominance_margin_db=3,
        smooth_window=5,
        pad_ms=200,
        bridge_gap_ms=150,
    )
