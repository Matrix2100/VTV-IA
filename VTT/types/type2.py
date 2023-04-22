import queue
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import sys
import json

'''This script processes audio input from the microphone and displays the transcribed text.'''


def main():
    # list all audio devices known to your system
    print("Display input/output devices")
    print(sd.query_devices())

    # get the samplerate - this is needed by the Kaldi recognizer
    device_info = sd.query_devices(sd.default.device[0], 'input')
    samplerate = int(device_info['default_samplerate'])

    # display the default input device
    print("===> Initial Default Device Number:{} Description: {}".format(sd.default.device[0], device_info))

    # setup queue and callback function
    q = queue.Queue()

    def record_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    # build the model and recognizer objects.
    print("===> Build the model and recognizer objects.  This will take a few minutes.")
    # get path from project root
    model = Model(r"/VTT/model/vosk-model-pt-fb-v0.1.1-pruned")
    print("===> Model built successfully.")
    recognizer = KaldiRecognizer(model, samplerate)
    recognizer.SetWords(False)

    print("===> Begin recording. Press Ctrl+C to stop the recording ")
    try:
        with sd.RawInputStream(dtype='int16',
                               channels=1,
                               callback=record_callback):
            while True:
                # print("===> Waiting for input sound")
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    recognizer_result = recognizer.Result()
                    # convert the recognizer_result string into a dictionary
                    result_dict = json.loads(recognizer_result)
                    if not result_dict.get("text", "") == "":
                        print(recognizer_result)
                    else:
                        print("no input sound")

    except KeyboardInterrupt:
        print('===> Finished Recording')
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    main()
