#Service file to setup the audio for the application 
#Running a speaker function and microphone function in thread real time audio response between model and user
import logging
import os
import sounddevice as sd
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
FORMAT = np.int16
SAMPLE_WIDTH = 2  #bytes for int16 samples
CHUNK = 1024               #audio frames per buffer
SILENCE_FRAMES = 12        #number of consecutive silent frames → end of speech

#File paths -> persisting user recordings + model speech responses
_AUDIO_DIR = pathlib.Path(__file__).resolve().parent / "audio_dir"
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
_AUDIO_FILE = _AUDIO_DIR / "audio.wav"
_MODEL_AUDIO_FILE = _AUDIO_DIR / "model_response.wav"
_SPEAKER_CACHE = None

#Thread-safe locks for file access
_AUDIO_FILE_LOCK = threading.Lock()
_RESPONSE_FILE_LOCK = threading.Lock()

#Voice activity detector singleton with configurable sensitivity
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.05"))
voice_detector = vad.EnergyVAD(sample_rate=RATE, frame_length=25, frame_shift=20,
                energy_threshold=VAD_ENERGY_THRESHOLD, pre_emphasis=0.95)

#Global sandbox reference -> allows agents to use files/tools when available
_AGENT_SANDBOX = None
_AGENT_STATE = {
    "design": "",
    "implementation": "",
    "validation": ""
}


#Helper to allow application to register sandbox instance for agents
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
def input_stream(mic_stream, terminal_event, processing_queue, playback_event):
    """Thread 1: Captures audio from microphone and detects speech"""
    speech_buffer = []  #collect frames while user is talking
    silent_count = 0    #tracks how many silent frames have passed
    is_speaking = False
    segment_has_voice = False
    MAX_BUFFER_SIZE = 500  #prevent memory leak from very long speech
    suppressing_input = False

    logging.info("[THREAD 1] Input stream started - listening for speech")

    def finalize_segment(reason: str):
        nonlocal speech_buffer, silent_count, is_speaking, segment_has_voice
        if not speech_buffer or not segment_has_voice:
            speech_buffer.clear()
            silent_count = 0
            is_speaking = False
            segment_has_voice = False
            return
        logging.info(f"[THREAD 1] Finalizing speech segment ({reason}) with {len(speech_buffer)} frames")
        try:
            with _AUDIO_FILE_LOCK:
                with wave.open(str(_AUDIO_FILE), 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(SAMPLE_WIDTH)
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(speech_buffer))

            try:
                processing_queue.put("process", timeout=1.0)
                logging.info("[THREAD 1] Audio queued for agent processing")
            except queue.Full:
                logging.warning("[THREAD 1] Processing queue full, dropping speech segment")
        except Exception as e:
            logging.error(f"[THREAD 1] Failed to save audio file: {e}")
        finally:
            speech_buffer.clear()
            silent_count = 0
            is_speaking = False
            segment_has_voice = False

    while not terminal_event.is_set():
        playback_active = playback_event.is_set()
        if playback_active and not suppressing_input:
            suppressing_input = True
            logging.info("[THREAD 1] Agent speaking - pausing microphone capture")
        elif not playback_active and suppressing_input:
            suppressing_input = False
            logging.info("[THREAD 1] Agent done speaking - resuming microphone capture")

        try:
            # Read from sounddevice stream - returns (frames, overflowed)
            frames, overflowed = mic_stream.read(CHUNK)
            if overflowed:
                logging.warning("[THREAD 1] Audio input buffer overflow detected")
        except Exception as err:
            logging.error(f"[THREAD 1] Microphone read failed: {err}")
            time.sleep(0.01)
            continue

        if playback_active:
            speech_buffer.clear()
            silent_count = 0
            is_speaking = False
            segment_has_voice = False
            time.sleep(0.01)
            continue

        if frames is None or len(frames) == 0:
            continue
        # Ensure data is int16 format
        audio = np.asarray(frames, dtype=FORMAT).reshape(-1)
        #Simple echo suppression: drop frames that match recent speaker output
        if _SPEAKER_CACHE is not None and len(_SPEAKER_CACHE) == len(audio):
            if np.allclose(audio, _SPEAKER_CACHE, atol=1):
                logging.debug("[THREAD 1] Ignoring potential playback echo")
                continue
        data = audio.tobytes()

        vad_result = voice_detector(audio)  #VAD returns whether speech energy detected
        talking = bool(np.any(vad_result)) if isinstance(vad_result, (list, tuple, np.ndarray)) else bool(vad_result)

        if talking:
            if not is_speaking:
                is_speaking = True
                logging.info("[THREAD 1] Speech detected - starting recording")
            segment_has_voice = True
            speech_buffer.append(data)  #stash frame until user stops talking
            
            #Prevent buffer overflow from very long speech
            if len(speech_buffer) > MAX_BUFFER_SIZE:
                logging.warning(f"[THREAD 1] Speech buffer exceeded {MAX_BUFFER_SIZE} frames, forcing finalize")
                finalize_segment("buffer limit")
            
            silent_count = 0

        else:
            if is_speaking:
                silent_count += 1    #count silence until phrase considered done
                speech_buffer.append(data)

                if silent_count >= SILENCE_FRAMES:
                    finalize_segment("silence")

    logging.info("[THREAD 1] Input stream terminated")


#Agent processing thread - runs designer/programmer/validator workflow
def agent_processing_thread(terminal_event, processing_queue, response_queue):
    """Thread 2: Processes transcripts through designer -> programmer -> validator"""
    logging.info("[THREAD 2] Agent processing thread started")
    
    while not _should_terminate(terminal_event):
        try:
            # Wait for audio to process
            processing_queue.get(timeout=0.5)
            logging.info("[THREAD 2] Processing audio transcript...")
            
            # Run the full agent workflow
            response_text = audio_to_model()
            
            if response_text:
                logging.info(f"[THREAD 2] Agent workflow complete, generating TTS...")
                try:
                    # Generate TTS audio file
                    audio_path = audio_agent_string2audio(response_text, _MODEL_AUDIO_FILE)
                    # Queue the audio file for playback
                    response_queue.put(str(audio_path))
                    logging.info("[THREAD 2] TTS audio queued for playback")
                except Exception as e:
                    logging.error(f"[THREAD 2] TTS generation failed: {e}")
            else:
                logging.warning("[THREAD 2] No response text generated from agents")
                
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"[THREAD 2] Agent processing error: {e}")
    
    logging.info("[THREAD 2] Agent processing thread terminated")


#speaker stream function
def output_stream(speaker_stream, terminal_state, response_queue, playback_event):
    """Thread 3: Plays back TTS audio responses"""
    logging.info("[THREAD 3] Output stream started - ready to play responses")
    
    while not _should_terminate(terminal_state):
        try:
            # Wait for audio response to play (blocking with timeout)
            response_audio_path = response_queue.get(timeout=0.5)
            logging.info(f"[THREAD 3] Playing audio response: {response_audio_path}")
            playback_event.set()
            _play_audio_response(speaker_stream, response_audio_path, CHUNK)
            logging.info("[THREAD 3] Audio playback complete")
            playback_event.clear()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"[THREAD 3] Audio playback error: {e}")
            playback_event.clear()
    
    logging.info("[THREAD 3] Output stream terminated")


def _play_audio_response(speaker_stream, audio_file_path, audio_chunks):
    """Helper to play audio response from file."""
    global _SPEAKER_CACHE
    try:
        with _RESPONSE_FILE_LOCK:
            with wave.open(audio_file_path, 'rb') as wav_reader:
                data = wav_reader.readframes(audio_chunks)
                while data:
                    frame_array = np.frombuffer(data, dtype=np.int16)
                    if CHANNELS == 1:
                        frame_array = frame_array.reshape(-1, 1)
                    else:
                        frame_array = frame_array.reshape(-1, CHANNELS)
                    speaker_stream.write(frame_array)
                    _SPEAKER_CACHE = frame_array.reshape(-1).copy()
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
    
    #Handle OpenAI response objects
    if hasattr(payload, 'choices'):
        if payload.choices and hasattr(payload.choices[0], 'message'):
            msg = payload.choices[0].message
            text = msg.content if hasattr(msg, 'content') else ""
        else:
            return ("", "")
    elif hasattr(payload, 'content'):
        text = payload.content or ""
    elif isinstance(payload, str):
        text = payload
    elif isinstance(payload, (list, tuple)):
        text = " ".join(str(item) for item in payload if item)
    else:
        text = str(payload or "")

    text = text.strip()
    if not text:
        return ("", "")
    
    if "/" not in text:
        return (text, "")

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
        shared_context = _build_shared_context()
        designer_prompt = transcript_text
        if shared_context:
            designer_prompt = (
                f"{transcript_text}\n\n"
                f"[Shared context from validator/programmer]\n{shared_context}"
            )
        designer_output = designer_agent(designer_prompt, sandbox=sandbox)
    except Exception as e:
        logging.exception(f"Designer agent failed to process input: {e}")
        workflow["speech"] = "I could not plan your request yet. Please try again."
        workflow["design_task"] = transcript_text
        return workflow

    designer_speech, programmer_task = _split_agent_sections(designer_output)
    workflow["design_task"] = programmer_task or transcript_text
    _update_agent_state("design", workflow["design_task"])
    spoken_parts = []
    task_summary = workflow["design_task"].strip() if workflow["design_task"] else ""
    if task_summary:
        spoken_parts.append(f"Here are the tasks I'll handle: {task_summary}")
    if designer_speech:
        spoken_parts.append(designer_speech.strip())
    workflow["speech"] = " ".join(spoken_parts) if spoken_parts else ""

    try:
        programmer_instructions = workflow["design_task"]
        prior_validation = _AGENT_STATE.get("validation")
        if prior_validation:
            programmer_instructions = (
                f"{programmer_instructions}\n\n"
                f"[Previous validator notes to address]\n{prior_validation}"
            )
        code_result = programmer_agent(programmer_instructions, sandbox=sandbox)
        workflow["code_result"] = code_result or ""
        _update_agent_state("implementation", workflow["code_result"])
    except Exception as e:
        logging.exception(f"Programmer agent failed to generate code: {e}")
        workflow["speech"] = (workflow["speech"] + " Programmer agent encountered an issue.").strip()
        return workflow

    try:
        validator_design_context = workflow["design_task"]
        designer_notes = _AGENT_STATE.get("design")
        if designer_notes and designer_notes != validator_design_context:
            validator_design_context = (
                f"{validator_design_context}\n\n[Designer summary]\n{designer_notes}"
            )
        validation_payload = validator_agent(validator_design_context, workflow["code_result"], sandbox=sandbox)
    except Exception as e:
        logging.exception(f"Validator agent failed to run checks: {e}")
        workflow["speech"] = (workflow["speech"] + " Validator agent could not validate the changes yet.").strip()
        return workflow

    validator_speech, validator_report = _split_agent_sections(validation_payload)
    workflow["validation_report"] = validator_report or str(validation_payload or "")
    _update_agent_state("validation", workflow["validation_report"])

    #Combining spoken responses from designer + validator
    spoken_parts = []
    if task_summary:
        spoken_parts.append(f"Here are the tasks I'll handle: {task_summary}")
    if designer_speech:
        spoken_parts.append(designer_speech.strip())
    if validator_speech:
        spoken_parts.append(validator_speech.strip())
    workflow["speech"] = (
        " ".join(spoken_parts)
        if spoken_parts else "I'll keep you updated as soon as I have a plan."
    )

    return workflow

#Function to run both functions in parallel -> real time audio stream between model and user 
#When request for audio conversation is pressed -> function is ran 
def parallel_audio_stream(terminal_state=None):
    """
    runs the functions in parallel allows for 
    conversing between model and user 
    """
    if isinstance(terminal_state, threading.Event):
        terminal_event = terminal_state
    else:
        terminal_event = threading.Event()
        if terminal_state:
            terminal_event.set()

    #Queue for coordinating between threads
    processing_queue = queue.Queue(maxsize=5)  # Input -> Agent processing
    response_queue = queue.Queue(maxsize=10)   # Agent processing -> Output
    playback_event = threading.Event()

    #Opening the two streams -> output and input 
    try:
        #microphone stream -> opens port to record audio to microphone
        microphone_stream = sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=CHUNK,
        )
        microphone_stream.start()

        #opening up the speaker stream -> opens the port to pass audio back from computer to user
        speaker_streaming = sd.OutputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=CHUNK,
        )
        speaker_streaming.start()
        
        logging.info("Audio streams opened successfully")
        
    except Exception as e:
        logging.error(f"Failed to open audio streams: {e}")
        logging.error("Check that your microphone/speakers are connected and not in use by another application")
        return terminal_event

    #Creating 3 threads for parallel processing
    thread1 = threading.Thread(
        target=input_stream, 
        args=(microphone_stream, terminal_event, processing_queue, playback_event),
        name="InputThread"
    )
    
    thread2 = threading.Thread(
        target=agent_processing_thread,
        args=(terminal_event, processing_queue, response_queue),
        name="AgentThread"
    )
    
    thread3 = threading.Thread(
        target=output_stream,
        args=(speaker_streaming, terminal_event, response_queue, playback_event),
        name="OutputThread"
    )

    # Start all threads
    thread1.start()
    thread2.start()
    thread3.start()
    
    logging.info("All threads started - system ready for voice interaction")

    try:
        while not terminal_event.is_set():
            time.sleep(0.1)  #keep parent thread alive until shutdown or ctrl+c
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down...")
        terminal_event.set()
    finally:
        #Wait for threads to finish with timeout
        logging.info("Waiting for threads to terminate...")
        thread1.join(timeout=5.0)
        thread2.join(timeout=5.0)
        thread3.join(timeout=5.0)
        
        #Force cleanup even if threads don't join cleanly
        try:
            microphone_stream.stop()
            microphone_stream.close() #closing the microphone stream -> no more audio feedback
            logging.info("Microphone stream closed")
        except Exception as e:
            logging.error(f"Error closing microphone stream: {e}")
        
        try:
            speaker_streaming.stop()
            speaker_streaming.close() #closes the speaker stream -> no more audio passed back
            logging.info("Speaker stream closed")
        except Exception as e:
            logging.error(f"Error closing speaker stream: {e}")

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

    logging.info(f"[TRANSCRIPT] {text}")

    #Running multi-agent workflow to keep designer/programmer/validator in sync
    workflow = _run_agent_workflow(text)

    if workflow["design_task"]:
        print(f"[DEBUG] Programmer Task -> {workflow['design_task']}")  #developer visibility
    if workflow["validation_report"]:
        print(f"[DEBUG] Validator Report -> {workflow['validation_report']}")

    return workflow["speech"]

#Legacy helper retained for tests -> reads chunk, returns (voice_detected, data)
def is_talking(audio_stream, audio_chunks):
    """Legacy test helper - reads audio and returns voice detection status"""
    try:
        frames, overflowed = audio_stream.read(audio_chunks)
        if overflowed:
            logging.warning("Audio overflow in is_talking")
        
        # Convert to int16 for VAD
        audio_sample = np.asarray(frames, dtype=np.int16).reshape(-1)
        
        raw = voice_detector(audio_sample)
        if isinstance(raw, (list, tuple, np.ndarray)):
            voice_detected = bool(np.any(raw))
        else:
            voice_detected = bool(raw)
        
        return (voice_detected, frames)
    except Exception as e:
        logging.error(f"Error in is_talking: {e}")
        return (False, np.zeros((audio_chunks, CHANNELS), dtype=np.int16))
def _build_shared_context() -> str:
    """Creates shared agent context describing recent implementation + validation."""
    sections = []
    implementation = _AGENT_STATE.get("implementation")
    validation = _AGENT_STATE.get("validation")
    if implementation:
        sections.append(f"Latest implementation summary:\n{implementation.strip()}")
    if validation:
        sections.append(f"Latest validation feedback:\n{validation.strip()}")
    return "\n\n".join(sections)


def _update_agent_state(key: str, value: str):
    """Stores latest agent communication for cross-agent context."""
    _AGENT_STATE[key] = (value or "").strip()
