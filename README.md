## Curant Service Runbook

### 1. Configure OpenAI credentials
```
export OPENAI_API_KEY="sk-your-key"
```
Set the environment variable in each shell (or load it from `server/.env`) so the designer/programmer/validator agents and audio helpers can authenticate with OpenAI.

### 2. Run automated checks
```
python3 -m compileall server/app/services
python3 server/tests/test_audio_service.py
```
The compile step catches syntax/import issues across all service modules; the unit test exercises the audio pipeline with mocked hardware/APIs.

### 3. Drive the multi-agent workflow without audio hardware
```
cd server
python3 run_agent_workflow.py "Summarize what the validator should check"
```
Add `--repo-url https://github.com/you/repo.git` if you want tool-enabled sessions inside the sandbox; the script will clone the repo, run designer→programmer→validator, print their outputs, and then clean up the sandbox container.

### 4. Run the live audio conversation (with optional sandbox)
```
cd server
python3 run_audio_agent.py --repo-url https://github.com/you/repo.git
```
Omit `--repo-url` if you just want a conversational demo; include it to let the agent read/write your repo through the sandbox while you speak. Use `Ctrl+C` to stop the audio loop.

### 5. Build/run the development container
```
docker build -t curant-dev -f docker/Dockerfile .
docker run -it --rm \
  --env-file server/.env \
  -p 8000:8000 \
  curant-dev
```
The Dockerfile installs all Python dependencies plus system packages required for PyAudio/ffmpeg so you can run the services consistently across machines.
