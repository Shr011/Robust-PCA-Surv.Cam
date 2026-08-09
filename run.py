# run.py  ← RUN THIS TO START

from realtime_pipeline import RealTimePipeline
from config import (LEARN_FRAMES, THRESHOLD,
                    MIN_AREA, REFRESH_MINS)

pipeline = RealTimePipeline(
    learn_frames = LEARN_FRAMES,
    threshold    = THRESHOLD,
    min_area     = MIN_AREA,
    refresh_mins = REFRESH_MINS
)

pipeline.run()