import torch
from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics

set_telemetry_metrics(False)  # opt out of the default phone-home

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="HF_TOKEN",  # your token here
)
pipeline.to(torch.device("mps"))

output = pipeline("your_clip.wav")            # no num_speakers set
diarization = output.speaker_diarization
predicted_count = len(diarization.labels())

print(f"Predicted speakers: {predicted_count}")
print(f"Labels: {diarization.labels()}")