import time


class FrameRateCounter:
    def __init__(self, smooth_alpha: float = 0.05):
        self.last_frame_time = time.perf_counter()
        self.smooth_alpha = smooth_alpha
        self.smooth_frame_time = -1.0

    def reset(self):
        self.last_frame_time = time.perf_counter()
        self.smooth_frame_time = -1.0

    def get_fps(self) -> float:
        return 1.0 / self.smooth_frame_time if self.smooth_frame_time > 0 else 0.0

    def update(self):
        now = time.perf_counter()
        frame_time = now - self.last_frame_time
        self.last_frame_time = now
        if self.smooth_frame_time < 0:
            self.smooth_frame_time = frame_time
        else:
            self.smooth_frame_time = self.smooth_alpha * frame_time + (1 - self.smooth_alpha) * self.smooth_frame_time
