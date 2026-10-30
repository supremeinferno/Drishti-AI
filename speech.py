from __future__ import annotations

import subprocess
import sys
import threading

ALERT_SOUND_MAC = "/System/Library/Sounds/Funk.aiff"


def beep():
    """A short alert that plays at once, before the spoken warning can start."""
    if sys.platform == "darwin":
        subprocess.Popen(["afplay", ALERT_SOUND_MAC])
    elif sys.platform == "win32":
        import winsound
        winsound.MessageBeep()
    else:
        print("\a", end="", flush=True)


class Speaker:
    """Speaks in a background thread so the video loop never freezes.

    Messages wait in two slots, urgent ("Careful!") and normal. Each slot keeps only the newest
    message, so stale guidance is dropped instead of piling up. An urgent message:
      - is always spoken before a normal one, and a normal message can never replace it
      - cuts off a normal message that is already being spoken
      - plays a beep immediately, so the user is warned even before the voice starts
    """

    def __init__(self, enabled: bool = True, rate: int = 175):
        self.enabled = enabled
        self.rate = rate
        self._cond = threading.Condition()
        self._urgent: str | None = None
        self._normal: str | None = None
        self._speaking_urgent = False
        self._busy = False                          # a message is being spoken right now
        self._proc: subprocess.Popen | None = None  # macOS `say` process now speaking
        self._engine = None                         # pyttsx3 engine on other platforms
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str, urgent: bool = False):
        print(f"[voice{' URGENT' if urgent else ''}] {text}")
        if not self.enabled:
            return
        with self._cond:
            if urgent:
                self._urgent = text
                if not self._speaking_urgent:
                    self._interrupt()
            else:
                self._normal = text
            self._cond.notify_all()
        if urgent:
            beep()

    def wait_until_quiet(self, timeout_s: float = 6.0):
        """Block until everything queued has been spoken, e.g. a last message before exiting."""
        if not self.enabled:
            return
        with self._cond:
            self._cond.wait_for(lambda: not self._busy and self._urgent is None
                                and self._normal is None, timeout_s)

    def _interrupt(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        elif self._engine is not None:
            self._engine.stop()  # pyttsx3: best effort, support varies by driver

    def _run(self):
        if sys.platform != "darwin":
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
        while True:
            with self._cond:
                while self._urgent is None and self._normal is None:
                    self._cond.wait()
                if self._urgent is not None:
                    text, self._urgent, self._speaking_urgent = self._urgent, None, True
                else:
                    text, self._normal, self._speaking_urgent = self._normal, None, False
                self._busy = True
                if self._engine is None:
                    # Started under the lock, so an urgent message can always find and stop it.
                    # pyttsx3 is unreliable off the main thread on macOS; the built-in `say` is not.
                    self._proc = subprocess.Popen(["say", "-r", str(self.rate), text])
            if self._engine is None:
                self._proc.wait()
            else:
                self._engine.say(text)
                self._engine.runAndWait()
            with self._cond:
                self._speaking_urgent = self._busy = False
                self._cond.notify_all()
