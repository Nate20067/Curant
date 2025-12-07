import json
from pathlib import Path

def load_tools(tool_json_path: str):
    """Loads a JSON tool schema from /tools folder and normalizes format"""
    tool_path = Path(tool_json_path)
    if not tool_path.exists():
        raise FileNotFoundError(f"Tool file not found: {tool_json_path}")
    
    with open(tool_path, "r") as f:
        data = json.load(f)
    
    # Normalize format - extract array if wrapped in {"tools": [...]}
    if isinstance(data, dict) and "tools" in data:
        return data["tools"]
    
    # Already in correct format (raw array)
    return data