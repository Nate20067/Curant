import os
import sys
import threading
import tempfile
import wave
import unittest
import types
import queue
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

#mocking external deps needed by audio_service
fake_pyaudio = types.ModuleType("pyaudio")


class _FakePyAudioInstance:
    def open(self, *_, **__):
        raise RuntimeError("PyAudio hardware not mocked in tests")

    def stop_stream(self):
        pass

    def close(self):
        pass

    def get_sample_size(self, _):
        return 2


fake_pyaudio.PyAudio = lambda: _FakePyAudioInstance()
fake_pyaudio.paInt16 = 2
sys.modules.setdefault("pyaudio", fake_pyaudio)

fake_vad = types.ModuleType("vad")


class _FakeEnergyVAD:
    def __init__(self, *_, **__):
        pass

    def __call__(self, *_args, **_kwargs):
        return False


fake_vad.EnergyVAD = _FakeEnergyVAD
sys.modules.setdefault("vad", fake_vad)

fake_openai = types.ModuleType("openai")


class _FakeChatCompletions:
    def create(self, *_, **__):
        message = types.SimpleNamespace(tool_calls=None, content="")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class _FakeChat:
    def __init__(self):
        self.completions = _FakeChatCompletions()


class _FakeAudioTranscriptions:
    def create(self, *_, **__):
        return types.SimpleNamespace(text="")


class _FakeStreamingResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def stream_to_file(self, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as wav_writer:
            wav_writer.setnchannels(1)
            wav_writer.setsampwidth(2)
            wav_writer.setframerate(16000)
            wav_writer.writeframes(b"\x00\x00" * 4)


class _FakeStreamingSpeech:
    def create(self, *_, **__):
        return _FakeStreamingResponse()


class _FakeAudioSpeech:
    def __init__(self):
        self.with_streaming_response = _FakeStreamingSpeech()


class _FakeAudio:
    def __init__(self):
        self.transcriptions = _FakeAudioTranscriptions()
        self.speech = _FakeAudioSpeech()


class _FakeOpenAIClient:
    def __init__(self, *_, **__):
        self.chat = _FakeChat()
        self.audio = _FakeAudio()


fake_openai.OpenAI = _FakeOpenAIClient
sys.modules.setdefault("openai", fake_openai)

fake_agents_pkg = types.ModuleType("agents")
fake_setup_agents = types.ModuleType("agents.setup_agents")


def _fake_audio_agent_audio2string(_path):
    return {}


class _FakeResponse:
    def __init__(self):
        self.content = ""


def _fake_deisgner_agent(_text):
    return _FakeResponse()


def _fake_audio_agent_string2audio(_text, output_path=None, response_format="wav"):
    target_path = Path(output_path or "fake_audio.wav")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target_path), "wb") as wav_writer:
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(16000)
        wav_writer.writeframes(b"\x00\x00" * 10)
    return target_path


fake_setup_agents.audio_agent_audio2string = _fake_audio_agent_audio2string
fake_setup_agents.deisgner_agent = _fake_deisgner_agent
fake_setup_agents.audio_agent_string2audio = _fake_audio_agent_string2audio
fake_agents_pkg.setup_agents = fake_setup_agents
sys.modules.setdefault("agents", fake_agents_pkg)
sys.modules.setdefault("agents.setup_agents", fake_setup_agents)

from app.services.audio_system import audio_service


class DummyMicrophone:
    def __init__(self, frames):
        self._frames = list(frames)
        self._last = frames[-1] if frames else b""

    def read(self, *_, **__):
        if self._frames:
            self._last = self._frames.pop(0)
            return self._last
        return self._last


class InputStreamTests(unittest.TestCase):
    def test_input_stream_writes_audio_when_voice_detected(self):
        #setting up temporary wav file location
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tmp_wav = tmp_path / "temp_audio.wav"

            original_audio_file = audio_service._AUDIO_FILE
            original_voice_detector = audio_service.voice_detector
            original_silence_frames = audio_service.SILENCE_FRAMES

            terminal_event = threading.Event()
            audio_service._AUDIO_FILE = tmp_wav
            audio_service.SILENCE_FRAMES = 1

            voice_results = [True, False]

            def fake_voice_detector(_):
                if voice_results:
                    result = voice_results.pop(0)
                    if not voice_results:
                        terminal_event.set()
                    return result
                terminal_event.set()
                return False

            audio_service.voice_detector = fake_voice_detector

            speech_frame = b"\x01\x02" * audio_service.CHUNK
            silence_frame = b"\x00\x00" * audio_service.CHUNK
            mic = DummyMicrophone([speech_frame, silence_frame])
            processing_queue = queue.Queue(maxsize=1)

            try:
                audio_service.input_stream(mic, terminal_event, processing_queue)
            finally:
                audio_service._AUDIO_FILE = original_audio_file
                audio_service.voice_detector = original_voice_detector
                audio_service.SILENCE_FRAMES = original_silence_frames

            self.assertTrue(tmp_wav.exists())

            with wave.open(str(tmp_wav), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), audio_service.RATE)
                self.assertGreater(wav_file.getnframes(), 0)


if __name__ == "__main__":
    unittest.main()
