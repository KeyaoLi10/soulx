import time


class TurnSession:
    def __init__(self, engine):
        self.engine = engine
        self.last_state = None

        # session lifecycle management
        now = time.time()
        self.created_ts = now
        self.last_active_ts = now

    def touch(self):
        """
        Called by server, used for session keep-alive / GC
        """
        self.last_active_ts = time.time()

    def feed_audio(self, audio):
        """
        audio: np.ndarray(float32)
        """
        # Consider active every time audio is received
        self.touch()

        result = self.engine.process(audio)
        self.last_state = result["state"]
        return result

    def set_max_wait_num(self, n):
        """Override the wait-path silence confirmation length (chunks).

        n: positive int, or None to fall back to the model config default.
        Takes effect on the very next fed chunk — no state reset, mid-turn
        changes only affect silence counted from then on (wait_idle_cnt is
        runtime state and is untouched here).
        """
        self.engine.max_wait_num = n

        # # Report only when state changes
        # if result["state"] != self.last_state:
        #     self.last_state = result["state"]
        #     return result

        # return None
