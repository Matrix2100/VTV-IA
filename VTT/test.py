import socket
from queue import Queue
from time import sleep

import speech_recognition as sr
import pyaudio

from socket_speech_recognition import SocketRecognizer, Options

HOST = 'localhost'
PORT = 4999

# Connect to the server
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn.connect((HOST, PORT))

# Set up the microphone emulator with desired options
options = Options(energy_threshold=300, # minimum audio energy to consider for recording
                  dynamic_energy_threshold=True,
                  dynamic_energy_adjustment_damping=0.15,
                  dynamic_energy_ratio=3,
                  pause_threshold=2, # seconds of non-speaking audio before a phrase is considered complete
                  operation_timeout=None, # seconds after an internal operation (e.g., an API request) starts before it times out, or ``None`` for no timeout
                  phrase_threshold=0.3, # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
                  non_speaking_duration=2) # seconds of non-speaking audio to keep on both sides of the recording

mic = SocketRecognizer(conn, options=options)
mic.adjust_for_ambient_noise(conn)
print("Connected to server")
# Set up the PyAudio output stream
p = pyaudio.PyAudio()
output_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
print("Listening...")
data_queue = Queue()


# Continuously read audio data from the microphone emulator
def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data_queue.put(audio.get_raw_data())


mic.listen_in_background(conn, record_callback, phrase_time_limit=5)
while True:
    if data_queue.empty():
        sleep(0.5)
    else:
        data = data_queue.get()
        output_stream.write(data)
