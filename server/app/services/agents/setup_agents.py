#Service file to setup the agents and connect them to the API key 
#Creating the Programmer, Validator and designer agents 
import base64
import binascii
import json
import logging
import os
import wave
from pathlib import Path
from typing import Iterable, Optional, Union

import openai
from elevenlabs.client import ElevenLabs

from .tool_loader import load_tools
from .agent_prompts import DESIGNER_PROMPT, PROGRAMMING_AGENT, VALIDATOR_PROMPT


TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"

#Creating functions to enact the agents workflow in the program 
#Creating the OpenAI client for all agents -> designer, programmer, validator
agent_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  #Getting API key from environment variables
)
_ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")
eleven_client = ElevenLabs(api_key=_ELEVENLABS_KEY) if _ELEVENLABS_KEY else None
ELEVEN_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
DEFAULT_ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "9BWtsMINqrJLrRacOk9x")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2")
ALLOWED_TTS_FORMATS = {
    "mp3_22050_32",
    "mp3_24000_48",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
    "pcm_8000",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_32000",
    "pcm_44100",
    "pcm_48000",
    "ulaw_8000",
    "alaw_8000",
    "opus_48000_32",
    "opus_48000_64",
    "opus_48000_96",
    "opus_48000_128",
    "opus_48000_192",
}
FORMAT_ALIASES = {
    "wav": "pcm_16000",
    "pcm": "pcm_16000",
    "pcm16": "pcm_16000",
    "default": "pcm_16000",
    "mp3": "mp3_44100_128",
}
RAW_DEFAULT_TTS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_TTS_FORMAT", "pcm_16000")
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2


def _resolve_voice_id() -> str:
    """Determines a valid ElevenLabs voice_id, accepting names or IDs."""
    candidate = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv("TTS_VOICE")
    #If candidate already looks like an ID (long, no spaces), use it directly
    if candidate and len(candidate) > 20 and " " not in candidate:
        return candidate

    if candidate and eleven_client:
        try:
            voices = eleven_client.voices.get_all()
        except Exception as exc:
            logging.warning("Unable to resolve ElevenLabs voice '%s': %s", candidate, exc)
        else:
            for voice in getattr(voices, "voices", []):
                name = getattr(voice, "name", "")
                if name and name.lower() == candidate.lower():
                    return voice.voice_id

    return DEFAULT_ELEVENLABS_VOICE_ID

#Creating the designer agent -> designs the plans to be sent to programmer based on user
def designer_agent(prompt: str, sandbox, conversation_history: list = None): 
    """Designer agent function -> creates design plans based on user prompt and writes design documents"""
    #Building conversation history -> starts with system context from designer
    messages = conversation_history or []
    
    #If no history exists -> adding initial designer context as system message
    if not messages:
        messages = [
            {'role': 'system', 'content': DESIGNER_PROMPT},  #Designer system prompt
            {'role': 'user', 'content': prompt}  #User provides their request
        ]
    else:
        #If history exists -> just appending new user message
        messages.append({'role': 'user', 'content': prompt})

    #Loading designer tools JSON
    tools = load_tools(TOOLS_DIR / "designer_tool.json")

    #Designer tool execution loop -> allows designer to update design docs
    while True:
        response = agent_client.chat.completions.create(
            model="gpt-4o",  #Using GPT-4o model for designer agent
            max_tokens=600,
            messages=messages,   #Passing full conversation history
            tools=tools,         #Passing designer tools
            tool_choice="auto"   #Allowing designer to call tools automatically
        )

        msg = response.choices[0].message

        #If designer returns normal output (no tool call)
        if not msg.tool_calls:
            messages.append({'role': 'assistant', 'content': msg.content})
            return msg.content  #Returning design output to system

        #Adding assistant message with tool calls
        messages.append({
            'role': 'assistant',
            'content': msg.content,
            'tool_calls': msg.tool_calls
        })

        #Handling tool calls from designer agent
        for call in msg.tool_calls:
            func_name = call.function.name
            args = json.loads(call.function.arguments)

            #Mapping tool call to sandbox method
            try:
                method = getattr(sandbox, func_name)
                result = method(**args)
            except AttributeError:
                result = {"error": f"Tool {func_name} not found in sandbox"}
            except Exception as e:
                result = {"error": str(e)}

            #Returning tool result back into conversation
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })

#Creating the programmer agent -> takes designer instructions and writes the code
def programmer_agent(design_instructions: str, sandbox, conversation_history: list = None):
    """Programmer agent function -> writes code based on designer instructions and uses tools"""
    #Building conversation history -> starts with programmer context
    messages = conversation_history or []
    
    #If no history exists -> adding initial programmer context as system message
    if not messages:
        messages = [
            {'role': 'system', 'content': PROGRAMMING_AGENT},  #Programmer system prompt
            {'role': 'user', 'content': design_instructions}  #Designer provides instructions
        ]
    else:
        #If history exists -> just appending new user message
        messages.append({'role': 'user', 'content': design_instructions})

    #Loading tools for programmer agent
    tools = load_tools(TOOLS_DIR / "programmer_tool.json")

    #Loop -> allows agent to call tools multiple times
    while True:
        response = agent_client.chat.completions.create(
            model="gpt-4o",  #Using GPT-4o model for programmer agent
            max_tokens=1200,  #Moderate token limit for code generation
            messages=messages,  #Passing full conversation history
            tools=tools,       #Passing programmer tools
            tool_choice="auto" #Allow model to call tools automatically
        )

        msg = response.choices[0].message

        #Checking if programmer agent returned normal content (no tool call)
        if not msg.tool_calls:
            messages.append({'role': 'assistant', 'content': msg.content})
            return msg.content  #Returning final result to application

        #Adding assistant message with tool calls
        messages.append({
            'role': 'assistant',
            'content': msg.content,
            'tool_calls': msg.tool_calls
        })

        #Handling tool calls from programmer agent
        for call in msg.tool_calls:
            func = call.function.name               #Tool function name
            args = json.loads(call.function.arguments)  #Tool arguments loaded from JSON

            #Calling Sandbox method that matches tool function name
            try:
                method = getattr(sandbox, func)
                result = method(**args)
            except AttributeError:
                result = {"error": f"Tool {func} not found in sandbox"}
            except Exception as e:
                result = {"error": str(e)}

            #Adding tool result back into conversation for agent to continue
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })



#Creating the validator agent -> reviews design and code to validate quality
def validator_agent(design: str, code: str, sandbox, conversation_history: list = None):
    """Validator agent function -> validates code, runs tests, checks similarity, and uses tools"""
    #Building conversation history -> starts with validator context
    messages = conversation_history or []
    
    #Creating validation prompt with design and code
    validation_prompt = f"Design Requirements:\n{design}\n\nGenerated Code:\n{code}\n\nPlease validate if the code meets the design requirements."
    
    #If no history exists -> adding initial validator context as system message
    if not messages:
        messages = [
            {'role': 'system', 'content': VALIDATOR_PROMPT},  #Validator system prompt
            {'role': 'user', 'content': validation_prompt}  #Providing design and code to validate
        ]
    else:
        #If history exists -> just appending new user message
        messages.append({'role': 'user', 'content': validation_prompt})

    #Loading tools for validator agent
    tools = load_tools(TOOLS_DIR / "validator_tool.json")

    #Loop -> validator may call multiple tools during validation
    while True:
        response = agent_client.chat.completions.create(
            model="gpt-4o",  #Using GPT-4o model for validator agent
            max_tokens=600,  #Medium token limit for validation feedback
            messages=messages,  #Passing full conversation history
            tools=tools,       #Passing validator tools
            tool_choice="auto" #Allow model to call tools automatically
        )

        msg = response.choices[0].message

        #If validator responds normally (no tools were called)
        if not msg.tool_calls:
            messages.append({'role': 'assistant', 'content': msg.content})
            return msg.content  #Returning validation report

        #Adding assistant message with tool calls
        messages.append({
            'role': 'assistant',
            'content': msg.content,
            'tool_calls': msg.tool_calls
        })

        #Handling validator tool calls
        for call in msg.tool_calls:
            func = call.function.name               #Tool function name
            args = json.loads(call.function.arguments)  #Tool arguments

            #Calling Sandbox method via Python reflection
            try:
                method = getattr(sandbox, func)
                result = method(**args)
            except AttributeError:
                result = {"error": f"Tool {func} not found in sandbox"}
            except Exception as e:
                result = {"error": str(e)}

            #Returning tool result to validator agent
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })

#Function to read in audio convert to text 
def audio_agent_audio2string(audio_file: Path) -> str:
    """Transcribes audio file to text using ElevenLabs speech-to-text"""
    if not eleven_client:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")

    with open(audio_file, "rb") as audio_handle:
        transcription = eleven_client.speech_to_text.convert(
            file=audio_handle,
            model_id=ELEVEN_STT_MODEL,
            tag_audio_events=False,
            language_code="eng",
            diarize=False,
        )

    if hasattr(transcription, "text"):
        return transcription.text
    if isinstance(transcription, dict):
        return transcription.get("text", "")
    return str(transcription or "")


#Service function to take in string and convert it to audio
def _normalize_tts_format(format_hint: Optional[str]) -> str:
    """Maps user-provided format strings/aliases to ElevenLabs-supported values."""
    fmt = (format_hint or "").strip().lower()
    if not fmt:
        fmt = "pcm_16000"
    fmt = FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in ALLOWED_TTS_FORMATS:
        logging.warning("Unsupported ElevenLabs output_format '%s', defaulting to pcm_16000", format_hint)
        return "pcm_16000"
    return fmt


DEFAULT_TTS_OUTPUT_FORMAT = _normalize_tts_format(RAW_DEFAULT_TTS_OUTPUT_FORMAT)


def _coerce_audio_bytes(payload) -> bytes:
    """
    Normalizes ElevenLabs responses into raw audio bytes.
    Handles streaming iterables, base64 strings, and dict wrappers.
    """
    if payload is None:
        raise ValueError("Empty TTS response payload")

    if hasattr(payload, "read"):
        return payload.read()

    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)

    if isinstance(payload, str):
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return payload.encode("utf-8")

    if isinstance(payload, dict):
        for key in ("audio", "data", "bytes", "content", "chunk"):
            if key in payload:
                return _coerce_audio_bytes(payload[key])
        raise ValueError("TTS payload dict missing audio bytes")

    if isinstance(payload, Iterable):
        buffer = bytearray()
        for chunk in payload:
            chunk_bytes = _coerce_audio_bytes(chunk)
            if chunk_bytes:
                buffer.extend(chunk_bytes)
        if buffer:
            return bytes(buffer)
        raise ValueError("TTS iterable payload yielded no audio bytes")

    raise TypeError(f"Unsupported TTS payload type: {type(payload)!r}")


def _write_audio_file(data: bytes, target_path: Path) -> Path:
    """Writes audio bytes to path, wrapping raw PCM into WAV if needed."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        with open(target_path, "wb") as audio_file:
            audio_file.write(data)
        return target_path

    logging.debug("TTS payload missing WAV header, wrapping bytes as PCM16")
    with wave.open(str(target_path), "wb") as wav_writer:
        wav_writer.setnchannels(PCM_CHANNELS)
        wav_writer.setsampwidth(PCM_SAMPLE_WIDTH)
        wav_writer.setframerate(PCM_SAMPLE_RATE)
        wav_writer.writeframes(data)
    return target_path


def audio_agent_string2audio(
    string_text: str,
    output_path: Optional[Union[Path, str]] = None,
    response_format: str = DEFAULT_TTS_OUTPUT_FORMAT
) -> Path:
    """Converts text string to audio file using ElevenLabs TTS"""
    if not eleven_client:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")

    #Setting target path -> defaults to audio_dir if no path provided
    target_path = Path(output_path) if output_path else Path("services/audio_system/audio_dir/audio_speech.wav")
    #Creating parent directories if they don't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    #Generate the response via TTS API
    voice_id = _resolve_voice_id()
    output_format = _normalize_tts_format(response_format)
    convert_kwargs = {
        "voice_id": voice_id,
        "output_format": output_format,
        "text": string_text,
    }
    if ELEVENLABS_TTS_MODEL:
        convert_kwargs["model_id"] = ELEVENLABS_TTS_MODEL
    audio_bytes = eleven_client.text_to_speech.convert(**convert_kwargs)
    data = _coerce_audio_bytes(audio_bytes)
    return _write_audio_file(data, target_path)  #Returning path to generated audio file
