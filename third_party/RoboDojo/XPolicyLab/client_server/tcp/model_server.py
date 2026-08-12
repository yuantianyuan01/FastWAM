import socket
import threading
import traceback
from .utils import *
import time

from XPolicyLab.utils.process_data import decode_obs_images

class ModelServer:
    def __init__(self, model, host="localhost", port=None):
        self.model = model
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.wait_interval = 10
        self.client_threads = []

    def start(self):
        """Start the model server and listen for incoming client connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.settimeout(self.wait_interval)
        self.server_socket.listen(5)
        self.running = True

        print(f"🚀 Model server started on {self.host}:{self.port}")
        print("🔄 Server is waiting for client connections...")

        self._accept_connections()

    def stop(self):
        """Stop the server and clean up resources gracefully"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        for t in self.client_threads:
            t.join(timeout=1)
        print("🛑 Server has been stopped")

    def _accept_connections(self):
        """Accept and handle new client connections"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"✅ Client connected from {addr}")
                # Handle each client in a separate thread
                t = threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True)
                t.start()
                self.client_threads.append(t)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"⚠️ Error accepting connection: {e}")
                break

    def _handle_client(self, client_socket):
        """Process requests from a single client"""
        with client_socket:
            while self.running:
                try:
                    # Read message length header (4 bytes, big-endian)
                    len_bytes = client_socket.recv(4)
                    if not len_bytes:
                        print("🔌 Client disconnected")
                        break
                    msg_length = int.from_bytes(len_bytes, "big")

                    # Read the full message based on length
                    chunks = []
                    remaining = msg_length
                    while remaining > 0:
                        chunk = client_socket.recv(min(remaining, 4096))
                        if not chunk:
                            raise ConnectionError("Incomplete data received")
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    raw_msg = b"".join(chunks).decode("utf-8")
                    # Deserialize JSON to Python, reconstruct any numpy arrays
                    data = json_to_numpy(raw_msg)
                    # data = pickle.loads(raw_msg)

                    # Extract command and observation
                    cmd = data.get("cmd")
                    obs = data.get("obs")  # None if not provided

                    # Decode every payload-carrying command, matching the ws
                    # server: custom RPCs can ship camera frames too, and
                    # decode_obs_images leaves non-observation payloads alone.
                    if obs is not None:
                        obs = decode_obs_images(obs)

                    # Find corresponding model method
                    method = getattr(self.model, cmd, None)
                    if not callable(method):
                        raise AttributeError(f"No model method named '{cmd}'")
                    # Call method with or without obs
                    
                    st = time.monotonic()
                    result = method(obs) if obs is not None else method()
                    response = {"res": result}

                    # Serialize response and send back with length header
                    resp_bytes = numpy_to_json(response).encode("utf-8")
                    client_socket.sendall(len(resp_bytes).to_bytes(4, "big"))
                    client_socket.sendall(resp_bytes)

                except (ConnectionResetError, BrokenPipeError):
                    print("🔌 Client connection lost")
                    break
                except Exception as e:
                    err = f"Error handling request: {e}"
                    print(f"⚠️ {err}")
                    tb = traceback.format_exc()
                    error_resp = numpy_to_json({"error": err, "traceback": tb}).encode("utf-8")
                    client_socket.sendall(len(error_resp).to_bytes(4, "big"))
                    client_socket.sendall(error_resp)
                    break
