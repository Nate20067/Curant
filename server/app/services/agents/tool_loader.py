import json
from pathlib import Path

def load_tools(tool_json_path: str):
    """Loads a JSON tool schema from /tools folder"""
    tool_path = Path(tool_json_path)
    if not tool_path.exists():
        raise FileNotFoundError(f"Tool file not found: {tool_json_path}")
    
    with open(tool_path, "r") as f:
        return json.load(f)
