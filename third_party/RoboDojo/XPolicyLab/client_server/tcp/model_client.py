from .utils import *
import socket
import sys
import time

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _status(level, color, message):
    print(f"{color}{BOLD}[{level}]{RESET} {message}", flush=True)


# Cold-start budget: a policy server loading a large checkpoint has not opened
# its port yet, so the client sees connection refused and retries.
# 180 x 5 s = 15 min, matching the websocket client.
CONNECT_MAX_ATTEMPTS = 180
CONNECT_RETRY_DELAY_S = 5


class ModelClient:
    def __init__(self, host="localhost", port=9999, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = CONNECT_MAX_ATTEMPTS
        retry_delay = CONNECT_RETRY_DELAY_S

        while attempts < max_attempts:
            try:
                _status(
                    "CONNECTING",
                    BLUE,
                    f"legacy TCP policy server {self.host}:{self.port} "
                    f"(attempt {attempts + 1}/{max_attempts}, timeout={self.timeout}s)",
                )
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                _status("CONNECTED", GREEN, f"legacy TCP policy server connected: {self.host}:{self.port}")
                return
            except Exception as e:
                attempts += 1
                if self.sock:
                    self.sock.close()
                    self.sock = None
                if attempts < max_attempts:
                    _status("CONNECT-FAILED", YELLOW, f"attempt {attempts}/{max_attempts} failed: {e}")
                    if not sys.stdout.isatty():
                        # Avoid flooding redirected logs with carriage-return updates.
                        _status("RECONNECT", YELLOW, f"retrying legacy TCP connection in {retry_delay}s")
                        time.sleep(retry_delay)
                    else:
                        for left in range(retry_delay, 0, -1):
                            print(
                                f"\r{YELLOW}{BOLD}[RECONNECT]{RESET} retrying legacy TCP connection in {left:2d}s",
                                end="",
                                flush=True,
                            )
                            time.sleep(1)
                        print("\r" + " " * 96 + "\r", end="", flush=True)
                else:
                    _status("ERROR", RED, f"failed to connect to legacy TCP policy server {self.host}:{self.port}")
                    raise ConnectionError(f"Failed to connect to server after {max_attempts} attempts: {str(e)}")

    def _send(self, data):
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode("utf-8")

            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, "big"))
            self.sock.sendall(json_data)

        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _send_recv(self, data):
        """Send request and receive response with numpy array support"""
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode("utf-8")

            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, "big"))
            self.sock.sendall(json_data)
            # Receive and deserialize response
            response = self._recv_response()
            return response

        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _recv_response(self):
        """Receive response with numpy array reconstruction"""
        # Read response length
        
        len_data = self.sock.recv(4)

        if not len_data:
            raise ConnectionError("Connection closed by server")

        size = int.from_bytes(len_data, "big")

        # Read complete response
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)
        # Deserialize with numpy reconstruction
        return json_to_numpy(b"".join(chunks).decode("utf-8"))

    def call(self, func_name=None, obs=None, _reconnect_attempted=False):
        request = {"cmd": func_name, "obs": obs}
        try:
            response = self._send_recv(request)
        except ConnectionError:
            if _reconnect_attempted:
                raise
            _status(
                "RECONNECT",
                YELLOW,
                f"legacy TCP connection dropped during {func_name}; reconnecting and retrying once",
            )
            self._connect()
            return self.call(func_name=func_name, obs=obs, _reconnect_attempted=True)
        if "res" in response.keys():
            return response["res"]
        return None

    def close(self):
        """Close the connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
                _status("DISCONNECTED", YELLOW, "legacy TCP connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    ModelClient()