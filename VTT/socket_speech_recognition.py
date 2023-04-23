import audioop
import collections
import math
import struct
import threading
from socket import socket

import pyaudio
import numpy as np
from speech_recognition import WaitTimeoutError, AudioData


class Options:
    def __init__(self,
                 chunk_size=1024,
                 sample_rate=16000,
                 energy_threshold=300,
                 dynamic_energy_threshold=True,
                 dynamic_energy_adjustment_damping=0.15,
                 dynamic_energy_ratio=1.5,
                 pause_threshold=0.8,
                 operation_timeout=None,
                 phrase_threshold=0.3,
                 non_speaking_duration=0.5):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.dynamic_energy_threshold = dynamic_energy_threshold
        self.dynamic_energy_adjustment_damping = dynamic_energy_adjustment_damping
        self.dynamic_energy_ratio = dynamic_energy_ratio
        self.pause_threshold = pause_threshold
        self.operation_timeout = operation_timeout
        self.phrase_threshold = phrase_threshold
        self.non_speaking_duration = non_speaking_duration


class SocketRecognizer:
    def __init__(self, conn, options):
        assert isinstance(conn, socket)
        assert isinstance(options, Options)

        self.conn = conn
        self.options = options

        self.pyaudio_module = self.get_pyaudio()
        self.format = self.pyaudio_module.paInt16  # 16-bit int sampling
        self.SAMPLE_WIDTH = self.pyaudio_module.get_sample_size(self.format)
        self.SAMPLE_RATE = self.options.sample_rate
        self.CHUNK_SIZE = self.options.chunk_size

        self.energy_threshold = self.options.energy_threshold
        self.dynamic_energy_threshold = self.options.dynamic_energy_threshold
        self.dynamic_energy_adjustment_damping = self.options.dynamic_energy_adjustment_damping
        self.dynamic_energy_ratio = self.options.dynamic_energy_ratio
        self.pause_threshold = self.options.pause_threshold
        self.operation_timeout = self.options.operation_timeout
        self.phrase_threshold = self.options.phrase_threshold
        self.non_speaking_duration = self.options.non_speaking_duration

        self.non_speaking_buffer = []

        self.adjustment_time = 0.5  # seconds
        self.quiet_duration = 0.5  # seconds

        self.energy_threshold_multiplier = 1.0
        self.adjustment_damping = self.dynamic_energy_adjustment_damping
        self.adjustment_rate = 1.5

        self.counter = 0

    @staticmethod
    def get_pyaudio():
        """
        Imports the pyaudio module and checks its version. Throws exceptions if pyaudio can't be found or a wrong version is installed
        """
        try:
            import pyaudio
        except ImportError:
            raise AttributeError("Could not find PyAudio; check installation")
        from distutils.version import LooseVersion
        if LooseVersion(pyaudio.__version__) < LooseVersion("0.2.11"):
            raise AttributeError("PyAudio 0.2.11 or later is required (found version {})".format(pyaudio.__version__))
        return pyaudio

    def listen(self, conn, timeout=None, phrase_time_limit=None):
        """
        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance, which it returns.

        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold`` (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold`` seconds of non-speaking or there is no more audio input. The ending silence is not included.

        The ``timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``, there will be no wait timeout.

        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, there will be no phrase time limit.

        This operation will always complete within ``timeout + phrase_timeout`` seconds if both are numbers, either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
        """
        # assert isinstance(source, AudioSource), "Source must be an audio source"
        # assert source.stream is not None, "Audio source must be entered before listening, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        # assert self.pause_threshold >= self.non_speaking_duration >= 0

        assert isinstance(conn, socket), "Source must be an socket connection"
        assert conn.recv(
            self.CHUNK_SIZE) is not None, "Audio source must be entered before listening, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.options.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = float(self.CHUNK_SIZE) / self.SAMPLE_RATE
        pause_buffer_count = int(math.ceil(
            self.pause_threshold / seconds_per_buffer))  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count = int(math.ceil(
            self.phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count = int(math.ceil(
            self.non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0  # number of seconds of audio read
        buffer = b""  # an empty buffer means that the stream has ended and there is no data left to read
        while True:
            frames = collections.deque()

            print("listening")
            # store audio input until the phrase starts
            while True:
                # handle waiting too long for phrase by raising an exception
                elapsed_time += seconds_per_buffer
                if timeout and elapsed_time > timeout:
                    raise WaitTimeoutError("listening timed out while waiting for phrase to start")

                buffer = conn.recv(self.options.chunk_size)
                print("len(buffer)", len(buffer))
                if len(buffer) == 0:
                    break  # reached end of the stream
                frames.append(buffer)
                if len(frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
                    frames.popleft()

                # detect whether speaking has started on audio input
                energy = audioop.rms(buffer, self.SAMPLE_WIDTH)  # energy of the audio signal
                print("energy", energy, "threshold", self.energy_threshold)
                if energy > self.energy_threshold:
                    break

                # dynamically adjust the energy threshold using asymmetric weighted average
                if self.dynamic_energy_threshold:
                    damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
                    target_energy = energy * self.dynamic_energy_ratio
                    self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            phrase_start_time = elapsed_time
            while True:
                # handle phrase being too long by cutting off the audio
                elapsed_time += seconds_per_buffer
                if phrase_time_limit and elapsed_time - phrase_start_time > phrase_time_limit:
                    break

                # buffer = source.stream.read(source.CHUNK)
                buffer = conn.recv(self.options.chunk_size)
                if len(buffer) == 0:
                    break  # reached end of the stream
                frames.append(buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                energy = audioop.rms(buffer, self.SAMPLE_WIDTH)  # unit energy of the audio signal within the buffer

                if energy > self.options.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:  # end of the phrase
                    break

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0:
                break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data
        for i in range(pause_count - non_speaking_buffer_count):
            frames.pop()  # remove extra non-speaking frames at the end
        frame_data = b"".join(frames)

        return AudioData(frame_data, self.SAMPLE_RATE, self.SAMPLE_WIDTH)

    def listen_in_background(self, conn, callback, phrase_time_limit=None):
        """
        Spawns a thread to repeatedly record phrases from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance and call ``callback`` with that ``AudioData`` instance as soon as each phrase are detected.

        Returns a function object that, when called, requests that the background listener thread stop. The background thread is a daemon and will not stop the program from exiting if there are no other non-daemon threads. The function accepts one parameter, ``wait_for_stop``: if truthy, the function will wait for the background listener to stop before returning, otherwise it will return immediately and the background listener thread might still be running for a second or two afterwards. Additionally, if you are using a truthy value for ``wait_for_stop``, you must call the function from the same thread you originally called ``listen_in_background`` from.

        Phrase recognition uses the exact same mechanism as ``recognizer_instance.listen(source)``. The ``phrase_time_limit`` parameter works in the same way as the ``phrase_time_limit`` parameter for ``recognizer_instance.listen(source)``, as well.

        The ``callback`` parameter is a function that should accept two parameters - the ``recognizer_instance``, and an ``AudioData`` instance representing the captured audio. Note that ``callback`` function will be called from a non-main thread.
        """
        assert isinstance(conn, socket), "Source must be an audio source"
        running = [True]

        def threaded_listen():
            while running[0]:
                try:  # listen for 1 second, then check again if the stop function has been called
                    audio = self.listen(conn, 1, phrase_time_limit)
                except WaitTimeoutError:  # listening timed out, just try again
                    pass
                else:
                    if running[0]:
                        callback(self, audio)

        def stopper(wait_for_stop=True):
            running[0] = False
            if wait_for_stop:
                listener_thread.join()  # block until the background thread is done, which can take around 1 second

        listener_thread = threading.Thread(target=threaded_listen)
        listener_thread.daemon = True
        listener_thread.start()
        return stopper

    def adjust_for_ambient_noise(self, conn, duration=1):
        """
        Adjusts the energy threshold dynamically using audio from ``source`` (an ``AudioSource`` instance) to account for ambient noise.

        Intended to calibrate the energy threshold with the ambient energy level. Should be used on periods of audio without speech - will stop early if any speech is detected.

        The ``duration`` parameter is the maximum number of seconds that it will dynamically adjust the threshold for before returning. This value should be at least 0.5 in order to get a representative sample of the ambient noise.
        """
        assert isinstance(conn, socket), "Source must be an audio source"
        assert conn.recv(
            self.CHUNK_SIZE) is not None, "Audio source must be entered before adjusting, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0, "Pause threshold must be at least as large as non-speaking duration"

        seconds_per_buffer = (self.CHUNK_SIZE + 0.0) / self.SAMPLE_RATE
        elapsed_time = 0

        # adjust energy threshold until a phrase starts
        while True:
            elapsed_time += seconds_per_buffer
            if elapsed_time > duration:
                break
            buffer = conn.recv(self.options.chunk_size)
            energy = audioop.rms(buffer, self.SAMPLE_WIDTH)  # energy of the audio signal

            # dynamically adjust the energy threshold using asymmetric weighted average
            damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
            target_energy = energy * self.dynamic_energy_ratio
            self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)
