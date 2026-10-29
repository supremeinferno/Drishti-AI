import queue
import subprocess
import sys
import threading


class Speaker:
    """Speaks in a background thread so the video loop never freezes.

    Only the most recent message is kept: if the user is still hearing the last one,
    stale messages are dropped instead of piling up.
    """

    def __init__(self, enabled: bool = True, rate: int = 175):
        self.enabled = enabled
        self.rate = rate
        self._queue: queue.Queue[str] = queue.Queue(maxsize=1)
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str):
        print(f"[voice] {text}")
        if not self.enabled:
            return
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        engine = None
        if sys.platform != "darwin":
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        while True:
            text = self._queue.get()
            if engine:
                engine.say(text)
                engine.runAndWait()
            else:
                # pyttsx3 is unreliable off the main thread on macOS; the built-in `say` is not.
                subprocess.run(["say", "-r", str(self.rate), text])
