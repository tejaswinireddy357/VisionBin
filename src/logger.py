import json
import os
from datetime import datetime

LOG_FILE = "logs/detection_log.json"

def ensure_log_file():
    """Create log file if it doesn't exist."""
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

def log_detection(waste_type, confidence):
    """Save a detection to the log file."""
    ensure_log_file()
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    logs.append({
        "waste_type":  waste_type,
        "confidence":  confidence,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date":        datetime.now().strftime("%Y-%m-%d"),
        "time":        datetime.now().strftime("%H:%M:%S"),
    })

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def get_logs():
    """Get all detection logs."""
    ensure_log_file()
    with open(LOG_FILE, "r") as f:
        return json.load(f)

def get_statistics():
    """Get count of each waste type detected."""
    logs = get_logs()
    stats = {}
    for log in logs:
        wt = log["waste_type"]
        stats[wt] = stats.get(wt, 0) + 1
    return stats

def clear_logs():
    """Clear all detection logs."""
    with open(LOG_FILE, "w") as f:
        json.dump([], f)