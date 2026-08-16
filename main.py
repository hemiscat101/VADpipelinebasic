import subprocess
import tempfile
import os

def parse_rttm(rttm_path, speaker_label):
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            spk = parts[7]
            if spk != speaker_label:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            segments.append((start, start + dur))
    segments.sort(key=lambda x: x[0])
    return segments


def run_sox(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sox failed: {result.stderr}")


def extract_speaker_audio(rttm_path, speaker_label, source_audio_path, output_path,
                           sox_path="sox", batch_size=100):
    segments = parse_rttm(rttm_path, speaker_label)
    if not segments:
        raise ValueError(f"No segments found for speaker '{speaker_label}' in {rttm_path}")

    print(f"{speaker_label}: {len(segments)} segments found")

    tmp_dir = tempfile.mkdtemp(prefix="sox_segs_")
    segment_files = []
    batch_files = []

    try:
        for i, (start, end) in enumerate(segments):
            dur = end - start
            seg_path = os.path.join(tmp_dir, f"seg_{i:05d}.wav")
            cmd = [sox_path, source_audio_path, seg_path, "trim", f"{start:.3f}", f"{dur:.3f}"]
            run_sox(cmd)
            segment_files.append(seg_path)

        
        for b, i in enumerate(range(0, len(segment_files), batch_size)):
            batch = segment_files[i:i + batch_size]
            batch_out = os.path.join(tmp_dir, f"batch_{b:04d}.wav")
            run_sox([sox_path] + batch + [batch_out])
            batch_files.append(batch_out)
            print(f"  batch {b}: merged {len(batch)} segments")

        
        run_sox([sox_path] + batch_files + [output_path])

        print(f"Combined audio for {speaker_label} written to {output_path}")

    finally:
        for f in segment_files + batch_files:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(tmp_dir)


if __name__ == "__main__":
    extract_speaker_audio(
        rttm_path=r"D:\VADbasic\fourspeaker.rttm",
        speaker_label="siddMTech",
        source_audio_path=r"D:\VADbasic\fourspeaker3pm BCM.wav",
        output_path=r"D:\VADBASIC\siddMTech.wav",
    )