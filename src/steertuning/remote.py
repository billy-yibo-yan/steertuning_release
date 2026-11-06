"""Connections to other computers and to Gemini, with timeouts on everything that can hang."""

from __future__ import annotations

import os
import queue
import threading

RPC_TIMEOUT_MS = 5000
RECEIVE_TIMEOUT_MS = 1000
GEMINI_TIMEOUT_MS = 120000  # one request, thinking included

# pyzmq can hang when the garbage collector takes a context and its socket in one pass, so clients stay alive.
_rpc_clients: list = []


def connect_rpc(host: str, port: int, timeout_ms: int = RPC_TIMEOUT_MS):
    """A commlink RPC client whose calls raise after a timeout instead of waiting on a dead server."""
    import commlink
    import zmq

    client = commlink.RPCClient(host=host, port=port)
    client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    client.socket.setsockopt(zmq.REQ_RELAXED, 1)  # the next call can go out after a timeout
    client.socket.setsockopt(zmq.REQ_CORRELATE, 1)  # and the late reply is dropped
    client.socket.setsockopt(zmq.LINGER, 0)  # closing never waits on a dead server
    _rpc_clients.append(client)
    return client


class TopicReader:
    """The latest message of one commlink topic, read in a daemon thread with a timeout."""

    def __init__(self, host: str, port: int, topic: str) -> None:
        self.host, self.port, self.topic = host, port, topic
        self._subscriber = self._subscribe()
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        self._reading = False

    def _subscribe(self):
        import commlink
        import zmq

        subscriber = commlink.Subscriber(self.host, port=self.port, topics=[self.topic], buffer=False)
        subscriber._get_req_socket(self.topic)[0].setsockopt(zmq.RCVTIMEO, RECEIVE_TIMEOUT_MS)
        return subscriber

    def _read_once(self) -> None:
        import zmq

        while True:
            try:
                self._results.put((self._subscriber.get(self.topic), None))
                return
            except zmq.Again:
                # A socket that timed out would serve the replies of its old requests, so read on a new one.
                self._subscriber.stop()
                self._subscriber = self._subscribe()
            except Exception as error:
                self._results.put((None, error))
                return

    def read(self, timeout: float = 2.0):
        """The newest message. Raises TimeoutError when none arrives in time."""
        if not self._reading:
            self._reading = True
            threading.Thread(target=self._read_once, daemon=True).start()
        try:
            message, error = self._results.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"No message on topic {self.topic} for {timeout:g} s.") from None
        self._reading = False
        if error is not None:
            raise error
        return message


def gemini_client():
    """A Gemini client with the key from the GEMINI_API_KEY environment variable. A stalled call raises."""
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable to use Gemini.")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
