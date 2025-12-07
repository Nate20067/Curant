#Service file to setup the agents and connect them to the API key 
#Creating the Programmer, Validator and designer agents 
import os
import openai
from pathlib import Path
from typing import Optional, Union
import json
from .tool_loader import load_tools
from .agent_prompts import DESIGNER_PROMPT, PROGRAMMING_AGENT, VALIDATOR_PROMPT


TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"

#Creating functions to enact the agents workflow in the program 
#Creating the OpenAI client for all agents -> designer, programmer, validator
agent_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  #Getting API key from environment variables
)

#Creating the designer agent -> designs the plans to be sent to programmer based on user
def designer_agent(prompt: str, sandbox, conversation_history: list = None): 
    """Designer agent function -> creates design plans based on user prompt and writes design documents"""
    #Building conversation history -> starts with system context from designer
    messages = conversation_history or []
    
    #If no history exists -> adding initial designer context as assistant message
    if not messages:
        messages = [
            {'role': 'assistant', 'content': DESIGNER_PROMPT},  #Designer introduces itself
            {'role': 'user', 'content': prompt}  #User provides their request
        ]
    else:
        #If history exists -> just appending new user message
        messages.append({'role': 'user', 'content': prompt})

    #Loading designer tools JSON
    tools = load_tools("tools/designer_tools.json")

    #Designer tool execution loop -> allows designer to update design docs
    while True:
        response = agent_client.chat.completions.create(
            model="gpt-5.1",  #Using GPT-5.1 model for designer agent
            max_tokens=800,
            messages=messages,   #Passing full conversation history
            tools=tools,         #Passing designer tools
            tool_choice="auto"   #Allowing designer to call tools automatically
        )

        msg = response.choices[0].message

        #If designer returns normal output (no tool call)
        if not msg.tool_calls:
            messages.append({'role': 'assistant', 'content': msg.content})
            return msg.content  #Returning design output to system

        #Handling tool calls from designer agent
        for call in msg.tool_calls:
            func_name = call.function.name
            args = json.loads(call.function.arguments)

            #Designer can only operate in design_docs/ folder, enforced here
            if "file" in args:
                file_path = f"design_docs/{args['file']}"
                args["file"] = file_path

            #Mapping tool call to sandbox method
            method = getattr(sandbox, func_name)
            result = method(**args)

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
    
    #If no history exists -> adding initial programmer context as assistant message
    if not messages:
        messages = [
            {'role': 'assistant', 'content': PROGRAMMING_AGENT},  #Programmer introduces itself
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
            model="gpt-5.1-codex-max",  #Using GPT-4o model for programmer agent
            max_tokens=2000,  #Higher token limit for code generation
            messages=messages,  #Passing full conversation history
            tools=tools,       #Passing programmer tools
            tool_choice="auto" #Allow model to call tools automatically
        )

        msg = response.choices[0].message

        #Checking if programmer agent returned normal content (no tool call)
        if not msg.tool_calls:
            messages.append({'role': 'assistant', 'content': msg.content})
            return msg.content  #Returning final result to application

        #Handling tool calls from programmer agent
        for call in msg.tool_calls:
            func = call.function.name               #Tool function name
            args = json.loads(call.function.arguments)  #Tool arguments loaded from JSON

            #Calling Sandbox method that matches tool function name
            method = getattr(sandbox, func)
            result = method(**args)

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
    
    #If no history exists -> adding initial validator context as assistant message
    if not messages:
        messages = [
            {'role': 'assistant', 'content': VALIDATOR_PROMPT},  #Validator introduces itself
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
            model="gpt-5.1",  #Using GPT-4o model for validator agent
            max_tokens=1000,  #Medium token limit for validation feedback
            messages=messages,  #Passing full conversation history
            tools=tools,       #Passing validator tools
            tool_choice="auto" #Allow model to call tools automatically
        )

        msg = response.choices[0].message

        #If validator responds normally (no tools were called)
        if not msg.tool_calls:
            messages.append({'role': 'assistant', 'content': msg.content})
            return msg.content  #Returning validation report

        #Handling validator tool calls
        for call in msg.tool_calls:
            func = call.function.name               #Tool function name
            args = json.loads(call.function.arguments)  #Tool arguments

            #Calling Sandbox method via Python reflection
            method = getattr(sandbox, func)
            result = method(**args)

            #Returning tool result to validator agent
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })

#Creating the audio client -> transcribes text to speech as well -> speech to text
#Creating a function to read the audio file and pass it to the model 
audio_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  #Getting API key from environment variables
) #Initializing audio client 

#Function to read in audio convert to text 
def audio_agent_audio2string(audio_file: Path) -> str:
    """Transcribes audio file to text using OpenAI Whisper model"""
    transcription = audio_client.audio.transcriptions.create(
        model="whisper-1",  #Using Whisper model for transcription
        file=audio_file,
        language="en"  #Setting language to English
    ) #Returning the transcription for the file -> returns string from audio file 
    return transcription.text  #Extracting just the text from response


#Service function to take in string and convert it to audio
def audio_agent_string2audio(string_text: str, output_path: Optional[Union[Path, str]] = None, response_format: str = "wav") -> Path:
    """Converts text string to audio file using OpenAI TTS model"""
    #Setting target path -> defaults to audio_dir if no path provided
    target_path = Path(output_path) if output_path else Path("services/audio_system/audio_dir/audio_speech.wav")
    #Creating parent directories if they don't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    #Streaming the audio response to file
    with audio_client.audio.speech.with_streaming_response.create(
        model="tts-1",  #Using TTS-1 model for speech generation
        voice="alloy",  #Using alloy voice -> can be changed to echo/fable/onyx/nova/shimmer
        response_format=response_format,  #Setting audio format (wav/mp3/opus/etc)
        input=string_text  #Text to be converted to speech
    ) as response:
        response.stream_to_file(str(target_path))  #Streaming response to target file
    
    return target_path  #Returning path to generated audio file
