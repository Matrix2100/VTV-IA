import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import cProfile

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import openai

import time

# recorder parameters
energy_threshold = 1600
dynamic_energy_threshold = False
record_timeout = 3
phrase_timeout = 3
dynamic_energy_adjustment_damping = 0.15
dynamic_energy_ratio = 1.5
pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete
phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
non_speaking_duration = 0.5

# whisper parameters
model = "large"
language = "Portuguese"

# microphone setting only for linux
if 'linux' in platform:
    default_microphone = 'pulse'


def microphone_source():
    if 'linux' in platform:
        mic_name = default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    return sr.Microphone(sample_rate=16000, device_index=index)
    else:
        return sr.Microphone(sample_rate=16000)


def get_recorder(source):
    recorder = sr.Recognizer()
    recorder.dynamic_energy_threshold = dynamic_energy_threshold
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_adjustment_damping = dynamic_energy_adjustment_damping
    recorder.dynamic_energy_ratio = dynamic_energy_ratio
    recorder.pause_threshold = pause_threshold
    recorder.phrase_threshold = phrase_threshold
    recorder.non_speaking_duration = non_speaking_duration
    with source:
        recorder.adjust_for_ambient_noise(source)
    return recorder


def main():
    print("Starting to setup microphone, recorder and queue...")
    source = microphone_source()
    recorder = get_recorder(source)
    data_queue = Queue()

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data_queue.put(audio.get_raw_data())
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    phrase_time = None
    last_sample = bytes()
    temp_file = NamedTemporaryFile().name
    print("Setup complete.")

    print("Loading model...")
    audio_model = whisper.load_model(model)
    print("Model loaded.")
    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                t4 = time.time()
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                t5 = time.time()
                print("Time to process audio: ", t5 - t4)

                t6 = time.time()
                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())
                t7 = time.time()
                print("Time to write to file: ", t7 - t6)

                # Read the transcription.
                t8 = time.time()

                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), language=language)
                text = result['text'].strip()
                t9 = time.time()
                print("Time to transcribe: ", t9 - t8)

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise, edit the existing one.
                if phrase_complete:
                    print(text)
                sleep(1)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
