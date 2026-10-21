"""Fixed-host, bounded worker-log snapshots; no CLI, proxies, redirects or retries."""
from __future__ import annotations

import http.client
import json
import re
import socket
import time
from datetime import datetime

from src.eval.assigned_smoke_control import ControlError

MAX_BYTES = 1_000_000


class WorkerLogReader:
    def __init__(self, endpoint, key, *, connection=None, seconds=5, quiet_seconds=0.75):
        if (not re.fullmatch(r"[a-z0-9]{1,64}", endpoint) or not key
                or not 0 < seconds <= 5 or not 0 < quiet_seconds <= 0.75):
            raise ValueError("bounded worker log endpoint and credential required")
        self.endpoint, self.key = endpoint, key
        self.seconds, self.quiet_seconds = seconds, quiet_seconds
        self.connection = connection or (
            lambda: http.client.HTTPSConnection("api.runpod.io", timeout=seconds))

    def read(self, worker):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", worker):
            raise ValueError("invalid worker log ID")
        started = time.monotonic()
        rows, buffer, data = [], b"", []
        connected, seen_bytes = False, 0
        last_frame = ""
        conn = self.connection()

        def output():
            return "".join(json.dumps(row) + "\n" for row in rows)

        def finish_frame():
            nonlocal last_frame
            if not data:
                return
            payload = b"\n".join(data).decode("utf-8")
            last_frame = payload[:4096]
            data.clear()
            row = json.loads(payload)
            if (not isinstance(row, dict) or row.get("source") not in {"container", "system"}
                    or not isinstance(row.get("line"), str)
                    or not isinstance(row.get("ts"), str) or row.get("truncated")
                    or row.get("workerId", worker) != worker):
                raise ValueError("invalid worker log record")
            if datetime.fromisoformat(row["ts"].replace("Z", "+00:00")).utcoffset() is None:
                raise ValueError("worker log timestamp needs a timezone")
            rows.append({"source": row["source"], "line": row["line"],
                         "ts": row["ts"], "workerId": worker})

        try:
            conn.request("GET", f"/v2/serverless/{self.endpoint}/workers/{worker}/logs?tail=5000",
                headers={"Authorization": "Bearer " + self.key,
                         "Accept": "text/event-stream", "User-Agent": "Qwen-Envoy/0.1"})
            remaining = self.seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError
            transport_socket = conn.sock
            transport_socket.settimeout(remaining)
            response = conn.getresponse()
            if response.status != 200:
                code = {401: "unauthorized", 403: "forbidden", 404: "not_found",
                        429: "rate_limited"}.get(response.status,
                        "server_error" if response.status >= 500 else "bad_response")
                raise ControlError("logs", code, response.status)
            if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != (
                    "text/event-stream"):
                raise ValueError("worker log response is not an event stream")
            connected = True
            while True:
                remaining = self.seconds - (time.monotonic() - started)
                if remaining <= 0:
                    break
                transport_socket.settimeout(min(remaining, self.quiet_seconds))
                chunk = response.read1(min(65536, MAX_BYTES + 1 - seen_bytes))
                if not chunk:
                    break
                seen_bytes += len(chunk)
                if seen_bytes > MAX_BYTES:
                    raise ValueError("worker log stream exceeds 1 MB")
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if not line:
                        finish_frame()
                    elif line.startswith(b"data:"):
                        value = line[5:]
                        data.append(value[1:] if value.startswith(b" ") else value)
                if len(rows) > 10000:
                    raise ValueError("worker log record limit exceeded")
            # An unterminated SSE frame is diagnostic data, never readiness evidence.
            return output()
        except (TimeoutError, socket.timeout):
            if connected:
                return output()
            raise ControlError("logs", "timeout", None, log_output=output()) from None
        except (OSError, http.client.HTTPException):
            raise ControlError("logs", "network_error", None, log_output=output()) from None
        except (ValueError, UnicodeError):
            diagnostic = json.dumps({"raw_incomplete_frame": last_frame or
                b"\n".join(data).decode("utf-8", errors="replace")[:4096]}) + "\n"
            raise ControlError("logs", "invalid_stream", None,
                               log_output=output() + diagnostic) from None
        finally:
            conn.close()
