#Service file to setup the audio for the application 
#Running a speaker function and microphone function in thread real time audio response between model and user
import logging
import pyaudio
import vad 
import threading
import time
import numpy as np
import pathlib
import wave
import queue
from concurrent.futures import ThreadPoolExecutor
from app.services.agents.setup_agents import (
    audio_agent_audio2string,
    audio_agent_string2audio,
    designer_agent,
    programmer_agent,
    validator_agent,
)


#Constants -> storing shared audio config for entire service
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024               #audio frames per buffer
SILENCE_FRAMES = 12        #number of consecutive silent frames → end of speech

#File paths -> persisting user recordings + model speech responses
_AUDIO_DIR = pathlib.Path(__file__).resolve().parent / "audio_dir"
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
_AUDIO_FILE = _AUDIO_DIR / "audio.wav"
_MODEL_AUDIO_FILE = _AUDIO_DIR / "model_response.wav"

#Thread-safe locks for file access
_AUDIO_FILE_LOCK = threading.Lock()
_RESPONSE_FILE_LOCK = threading.Lock()

#PyAudio + VAD -> reusable singletons for the mic + speaker threads
audio_object = pyaudio.PyAudio()
voice_detector = vad.EnergyVAD(sample_rate=RATE, frame_length=25, frame_shift=20,
                energy_threshold=0.10, pre_emphasis=0.95)

#Global sandbox reference -> allows agents to use files/tools when available
_AGENT_SANDBOX = None

#Thread pool for async agent processing -> prevents blocking audio loop
_AGENT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent_worker")


#Helper to allow application to register sandbox instance for agents
def configure_agent_sandbox(sandbox):
    """Registers sandbox instance for real tool calls."""
    global _AGENT_SANDBOX
    _AGENT_SANDBOX = sandbox

def _should_terminate(terminal_state) -> bool:
    """Normalize termination signal across bools and threading.Event."""
    if isinstance(terminal_state, threading.Event):
        return terminal_state.is_set()
    if callable(terminal_state):
        return bool(terminal_state())
    return bool(terminal_state)


#Function to read audio from the user with the input stream 
def input_stream(mic_stream, terminal_event, processing_queue):
    speech_buffer = []  #collect frames while user is talking
    silent_count = 0    #tracks how many silent frames have passed
    is_speaking = False
    MAX_BUFFER_SIZE = 500  #prevent memory leak from very long speech

    while not terminal_event.is_set():
        data = mic_stream.read(CHUNK, exception_on_overflow=False)  #pull raw bytes from mic
        audio = np.frombuffer(data, dtype=np.int16)                 #convert bytes to ints for VAD

        vad_result = voice_detector(audio)  #VAD returns whether speech energy detected
        talking = bool(np.any(vad_result)) if isinstance(vad_result, (list, tuple, np.ndarray)) else bool(vad_result)

        if talking:
            if not is_speaking:
                is_speaking = True
            speech_buffer.append(data)  #stash frame until user stops talking
            
            #Prevent buffer overflow from very long speech
            if len(speech_buffer) > MAX_BUFFER_SIZE:
                speech_buffer = speech_buffer[-MAX_BUFFER_SIZE:]
            
            silent_count = 0

        else:
            if is_speaking:
                silent_count += 1    #count silence until phrase considered done
                speech_buffer.append(data)

                if silent_count >= SILENCE_FRAMES:
                    #user has gone quiet long enough -> treat phrase as complete
                    #save audio buffer to file with thread-safe lock
                    try:
                        with _AUDIO_FILE_LOCK:
                            with wave.open(str(_AUDIO_FILE), 'wb') as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(audio_object.get_sample_size(FORMAT))
                                wf.setframerate(RATE)
                                wf.writeframes(b''.join(speech_buffer))
                        
                        #signal output thread to process (with timeout to prevent blocking)
                        try:
                            processing_queue.put("process", timeout=1.0)
                        except queue.Full:
                            logging.warning("Processing queue full, dropping speech segment")
                    except Exception as e:
                        logging.error(f"Failed to save audio file: {e}")

                    speech_buffer.clear()
                    silent_count = 0
                    is_speaking = False


#speaker stream function
def output_stream(speaker_stream, audio_chunks, terminal_state, processing_queue, response_queue):
    """
    Listens for audio processing signals and plays back model responses.
    Now uses async agent processing to avoid blocking the audio loop.
    """
    while not _should_terminate(terminal_state):
        #wait for signal that audio is ready to process
        try:
            processing_queue.get(timeout=0.5)
        except queue.Empty:
            #Check for pending responses even if no new audio
            try:
                response_audio_path = response_queue.get(block=False)
                _play_audio_response(speaker_stream, response_audio_path, audio_chunks)
            except queue.Empty:
                pass
            continue

        #Submit agent workflow to background thread (non-blocking!)
        _AGENT_EXECUTOR.submit(_async_agent_processing, response_queue)
        
        #Immediately check if there's a response ready to play
        try:
            response_audio_path = response_queue.get(timeout=0.1)
            _play_audio_response(speaker_stream, response_audio_path, audio_chunks)
        except queue.Empty:
            pass  #No response yet, continue listening


def _async_agent_processing(response_queue):
    """
    Runs agent workflow in background thread.
    Places generated audio file path into response_queue when ready.
    """
    try:
        response_text = audio_to_model()  #read transcript + designer reply
        if not response_text:
            return

        #Generate TTS audio file
        try:
            response_audio = audio_agent_string2audio(response_text, _MODEL_AUDIO_FILE)
            #Put the audio file path in queue for playback
            response_queue.put(str(_MODEL_AUDIO_FILE))
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
    except Exception as e:
        logging.error(f"Agent processing failed: {e}")


def _play_audio_response(speaker_stream, audio_file_path, audio_chunks):
    """Helper to play audio response from file."""
    try:
        with _RESPONSE_FILE_LOCK:
            with wave.open(audio_file_path, 'rb') as wav_reader:
                data = wav_reader.readframes(audio_chunks)  #read response file chunk by chunk
                while data:
                    speaker_stream.write(data)
                    data = wav_reader.readframes(audio_chunks)
    except (FileNotFoundError, wave.Error) as e:
        logging.error(f"Failed to play audio response: {e}")
    except Exception as e:
        logging.error(f"Unexpected error playing audio: {e}")


#Helper to split agent formatted responses "<speech>/<task>"
def _split_agent_sections(payload):
    """Splits agent response into spoken and technical halves."""
    if payload is None:
        return ("", "")

    text = ""
    if isinstance(payload, (list, tuple)):
        text = " ".join(str(item) for item in payload if item)
    else:
        text = str(payload or "")

    if not text.strip():
        return ("", "")

    if "/" not in text:
        return (text.strip(), "")

    left, right = text.split("/", 1)
    return (left.strip(), right.strip())


#Helper to run the multi-agent workflow for a transcript string
def _run_agent_workflow(transcript_text: str) -> dict:
    """Runs designer -> programmer -> validator cycle and returns conversation data."""
    sandbox = _AGENT_SANDBOX
    workflow = {
        "speech": "",
        "design_task": "",
        "code_result": "",
        "validation_report": "",
    }

    try:
        designer_output = designer_agent(transcript_text, sandbox=sandbox)
    except Exception as e:
        logging.exception(f"Designer agent failed to process input: {e}")
        workflow["speech"] = "I could not plan your request yet. Please try again."
        workflow["design_task"] = transcript_text
        return workflow

    designer_speech, programmer_task = _split_agent_sections(designer_output)
    workflow["design_task"] = programmer_task or transcript_text
    workflow["speech"] = designer_speech or ""

    try:
        code_result = programmer_agent(workflow["design_task"], sandbox=sandbox)
        workflow["code_result"] = code_result or ""
    except Exception as e:
        logging.exception(f"Programmer agent failed to generate code: {e}")
        workflow["speech"] = (workflow["speech"] + " Programmer agent encountered an issue.").strip()
        return workflow

    try:
        validation_payload = validator_agent(workflow["design_task"], workflow["code_result"], sandbox=sandbox)
    except Exception as e:
        logging.exception(f"Validator agent failed to run checks: {e}")
        workflow["speech"] = (workflow["speech"] + " Validator agent could not validate the changes yet.").strip()
        return workflow

    validator_speech, validator_report = _split_agent_sections(validation_payload)
    workflow["validation_report"] = validator_report or str(validation_payload or "")

    #Combining spoken responses from designer + validator
    spoken_parts = [part for part in [workflow["speech"], validator_speech] if part]
    workflow["speech"] = " ".join(spoken_parts) if spoken_parts else "Here is the latest project update."

    return workflow

#Function to run both functions in parrellel -> real time audio stream between model and user 
#When request for audio conversation is pressed -> function is ran 
def parallel_audio_stream(terminal_state=None):
    """
    runs the functions in parrellel allows for 
    conversing between model and user 
    """
    if isinstance(terminal_state, threading.Event):
        terminal_event = terminal_state
    else:
        terminal_event = threading.Event()
        if terminal_state:
            terminal_event.set()

    #Queue for coordinating between input and output threads
    processing_queue = queue.Queue(maxsize=5)
    #Queue for async agent responses (audio file paths)
    response_queue = queue.Queue(maxsize=10)

    #Opening the two streams -> output and input 
    #microphone stream -> opens port to record audio to microphone
    microphone_stream = audio_object.open(
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        output=False,
        frames_per_buffer=CHUNK
    )

    #opening up the speaker stream -> opens the port to pass audio back from computer to user
    speaker_streaming = audio_object.open(
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=False,
        output=True,
        frames_per_buffer=CHUNK
    )

    #Threading the functions for them to run in parrellel 
    thread1 = threading.Thread(target=input_stream, args=(microphone_stream, terminal_event, processing_queue))  #mic listener
    thread2 = threading.Thread(target=output_stream, args=(speaker_streaming, CHUNK, terminal_event, processing_queue, response_queue))  #speaker output

    thread1.start() #starting the input stream 
    thread2.start() #starting the output stream 

    try:
        while not terminal_event.is_set():
            time.sleep(0.1)  #keep parent thread alive until shutdown or ctrl+c
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down...")
        terminal_event.set()
    finally:
        #Wait for threads to finish with timeout
        thread1.join(timeout=5.0)
        thread2.join(timeout=5.0)
        
        #Force cleanup even if threads don't join cleanly
        try:
            microphone_stream.stop_stream()
            microphone_stream.close() #closing the microphone stream -> no more audio feedback
        except Exception as e:
            logging.error(f"Error closing microphone stream: {e}")
        
        try:
            speaker_streaming.stop_stream()
            speaker_streaming.close() #closes the speaker stream -> no more audio passed back
        except Exception as e:
            logging.error(f"Error closing speaker stream: {e}")
        
        #Shutdown executor
        _AGENT_EXECUTOR.shutdown(wait=False)

    return terminal_event


#helper function to pass the audio file to the model 
def audio_to_model():
    """
    Reads audio file, transcribes it, and runs agent workflow.
    Returns the speech response text for TTS.
    """
    #reading wav file data to audio agent with thread-safe lock
    with _AUDIO_FILE_LOCK:
        if not _AUDIO_FILE.exists():
            logging.warning("Audio file not found")
            return None

        try:
            transcript = audio_agent_audio2string(_AUDIO_FILE)
        except Exception as e:
            logging.error(f"Failed to transcribe audio: {e}")
            return None

    if isinstance(transcript, dict):
        text = transcript.get('text') or ''
    else:
        text = str(transcript or '').strip()
    
    if not text:
        logging.warning("Empty transcript received")
        return None

    #Running multi-agent workflow to keep designer/programmer/validator in sync
    workflow = _run_agent_workflow(text)

    if workflow["design_task"]:
        print(f"[DEBUG] Programmer Task -> {workflow['design_task']}")  #developer visibility
    if workflow["validation_report"]:
        print(f"[DEBUG] Validator Report -> {workflow['validation_report']}")

    return workflow["speech"]

#Legacy helper retained for tests -> reads chunk, returns (voice_detected, data)
def is_talking(audio_stream, audio_chunks):
    data = audio_stream.read(audio_chunks)
    audio_sample = np.frombuffer(data, dtype=np.int16)

    raw = voice_detector(audio_sample)
    if isinstance(raw, (list, tuple, np.ndarray)):
        voice_detected = bool(np.any(raw))
    else:
        voice_detected = bool(raw)

    return (voice_detected, data)