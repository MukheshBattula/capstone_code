import json
from datetime import datatime

LOG_FILE = "logs/detection_log.json"



def log_detection(file_name,status):
    """Logs flagged samples into a JSON file."""
   log_entry = {
        "timestamp": str(datetime.now()),
        "file_name": file_name,
        "status": status # poisen or not
    }

    try:
        with open(LOG_FILE,"r") as file:
            logs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as file:
        json.dump(logs, file, indent=4)
   
