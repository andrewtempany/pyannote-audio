import argparse

import torch
from dotenv import load_dotenv
import os
from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics

parser = argparse.ArgumentParser(description="Count speakers in an audio/video file.")
parser.add_argument("audio_path", help="Path to the audio or video file to diarize")
args = parser.parse_args()

load_dotenv()
token = os.environ["HF_TOKEN"]

set_telemetry_metrics(False)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token,
)
pipeline.to(torch.device("cuda"))

output = pipeline(args.audio_path)
diarization = output.speaker_diarization

print(f"Predicted speakers: {len(diarization.labels())}")
print(f"Labels: {diarization.labels()}")