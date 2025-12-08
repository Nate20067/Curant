import base64
import sys
import tempfile
import unittest
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agents import setup_agents


class CoerceAudioBytesTests(unittest.TestCase):
    def test_handles_iterable_byte_chunks(self):
        payload = [b"\x00\x01", b"\x02\x03"]
        result = setup_agents._coerce_audio_bytes(payload)
        self.assertEqual(result, b"\x00\x01\x02\x03")

    def test_handles_dict_with_base64_audio(self):
        audio = b"\x10\x20\x30"
        payload = {"audio": base64.b64encode(audio).decode("ascii")}
        result = setup_agents._coerce_audio_bytes(payload)
        self.assertEqual(result, audio)


class WriteAudioFileTests(unittest.TestCase):
    def test_raw_pcm_bytes_wrapped_as_wav(self):
        data = (b"\x00\x00\x10\x00" * 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.wav"
            setup_agents._write_audio_file(data, path)
            with wave.open(str(path), "rb") as wav_reader:
                self.assertEqual(wav_reader.getnchannels(), setup_agents.PCM_CHANNELS)
                self.assertEqual(wav_reader.getframerate(), setup_agents.PCM_SAMPLE_RATE)
                frames = wav_reader.readframes(5)
                self.assertTrue(frames)


class NormalizeFormatTests(unittest.TestCase):
    def test_alias_wav_maps_to_pcm(self):
        self.assertEqual(setup_agents._normalize_tts_format("wav"), "pcm_16000")

    def test_invalid_format_defaults(self):
        self.assertEqual(setup_agents._normalize_tts_format("foo"), "pcm_16000")
