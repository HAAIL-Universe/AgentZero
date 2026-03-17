import json
import time
import urllib.error
import urllib.request

status_url = "http://127.0.0.1:8888/api/status"
load_url = "http://127.0.0.1:8888/api/load-model"
deadline = time.time() + 7200
while time.time() < deadline:
    try:
        with urllib.request.urlopen(status_url, timeout=10) as response:
            payload = json.loads(response.read().decode('utf-8'))
        if payload.get('model', {}).get('status') == 'ready':
            break
        request = urllib.request.Request(load_url, data=b'', method='POST')
        with urllib.request.urlopen(request, timeout=3600):
            break
    except Exception:
        time.sleep(10)
