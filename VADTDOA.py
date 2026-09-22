import numpy as np
import librosa


def smooth_energy(energy_db, window=5):
    if window <= 1:
        return energy_db
    kernel = np.ones(window) / window
    return np.convolve(energy_db, kernel, mode="same")


def load_all_channels(speaker_files, sr=16000):
    audio = {}
    for spk, path in speaker_files.items():
        wav, _ = librosa.load(path, sr=sr, mono=True)
        audio[spk] = wav

    min_len = min(len(w) for w in audio.values())
    for spk in audio:
        audio[spk] = audio[spk][:min_len]

    return audio, sr


def frame_energy(wav, frame_len, hop_len):
    frames = librosa.util.frame(wav, frame_length=frame_len, hop_length=hop_len)
    energy = np.sqrt(np.mean(frames ** 2, axis=0))
    return energy


def per_channel_threshold(energy_db, floor_percentile=10, margin_db=12):
    noise_floor = np.percentile(energy_db, floor_percentile)
    return noise_floor + margin_db


def gcc_phat(sig, refsig, fs, max_tau=0.01, interp=1):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    R /= (np.abs(R) + 1e-15)
    cc = np.fft.irfft(R, n=interp * n)

    max_shift = int(interp * fs * max_tau)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(interp * fs)


def leading_speaker(window_audio, speakers, sr, max_tau=0.01):
    """Returns the speaker whose mic the sound reaches earliest (closest to
    the true talker), based on pairwise GCC-PHAT delay estimates."""
    lag_sum = {spk: 0.0 for spk in speakers}
    for i, spk_i in enumerate(speakers):
        for spk_j in speakers[i + 1:]:
            tau = gcc_phat(window_audio[spk_i], window_audio[spk_j], sr, max_tau=max_tau)
            lag_sum[spk_i] += tau
            lag_sum[spk_j] -= tau
    return min(lag_sum, key=lag_sum.get)


def verify_segments_with_tdoa(segments, waveforms, speakers, sr=16000,
                               max_tau=0.01, min_seg_for_tdoa=0.05):
    """Re-check each energy-derived segment against TDOA. Overrides the label
    whenever TDOA disagrees, to catch gain/bleed-driven mistakes.
    `waveforms` is the same dict already loaded by load_all_channels."""
    verified = []
    for spk, start, end in segments:
        s_idx, e_idx = int(start * sr), int(end * sr)

        if e_idx - s_idx < int(min_seg_for_tdoa * sr):
            verified.append((spk, start, end))
            continue

        window_audio = {s: waveforms[s][s_idx:e_idx] for s in speakers}
        tdoa_winner = leading_speaker(window_audio, speakers, sr, max_tau)
        verified.append((tdoa_winner, start, end))

    return verified


def energy_based_diarization(speaker_files, session_id, output_rttm,
                              sr=16000, frame_ms=25, hop_ms=10,
                              floor_percentile=10, silence_margin_db=12,
                              dominance_margin_db=3,
                              smooth_window=5, pad_ms=200, bridge_gap_ms=150,
                              use_tdoa=True, tdoa_max_tau=0.01):

    audio, sr = load_all_channels(speaker_files, sr=sr)
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    speakers = list(audio.keys())

    energies = np.stack([frame_energy(audio[spk], frame_len, hop_len) for spk in speakers])
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
    merged = []
    for spk, s, e in padded:
        if merged and merged[-1][0] == spk and (s - merged[-1][2]) <= bridge_s:
            merged[-1] = (merged[-1][0], merged[-1][1], max(merged[-1][2], e))
        else:
            merged.append((spk, s, e))
    segments = merged

    # ---- TDOA check, run as the final step on the finished segments ----
    if use_tdoa:
        segments = verify_segments_with_tdoa(
            segments, audio, speakers, sr=sr, max_tau=tdoa_max_tau
        )
        # re-merge in case TDOA relabeled two adjacent segments to the same speaker
        re_merged = []
        for spk, s, e in segments:
            if re_merged and re_merged[-1][0] == spk and abs(re_merged[-1][2] - s) < 1e-6:
                re_merged[-1] = (spk, re_merged[-1][1], e)
            else:
                re_merged.append((spk, s, e))
        segments = re_merged

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
        "nikhilLapel": r"/DATA/VAD/fourspeaker3pm nikhilLapel.wav",
        "paddyLapel": r"/DATA/VAD/fourspeaker3pm paddyLapel.wav",
        "siddMTechLapel": r"/DATA/VAD/fourspeaker3pm siddMTechLapel.wav",
        "siddPhdLapel": r"/DATA/VAD/fourspeaker3pm siddPhdLapel.wav"
    }

    energy_based_diarization(
        speaker_files=speaker_files,
        session_id="fourspeaker",
        output_rttm=r"D:\VADBASIC\4spk.rttm",
        floor_percentile=5,
        silence_margin_db=14,
        dominance_margin_db=3,
        smooth_window=5,
        pad_ms=200,
        bridge_gap_ms=150,
        use_tdoa=True,       
        tdoa_max_tau=0.01,  
    )