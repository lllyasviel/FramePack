import logging
import os
import random
import sys
from typing import Optional

import numpy as np
from pydantic import BaseModel


def setup_logging(rank: int = 0):
    """Setup logging configuration with rank information.
    
    Args:
        rank: The rank ID of the process. Defaults to 0 for non-distributed runs.
    """
    rank = int(os.environ.get("LOCAL_RANK", rank))
    logging.basicConfig(
        level=logging.INFO,
        format=f'[rank:{rank}] [%(levelname)s] %(asctime)s %(pathname)s:%(lineno)d %(message)s'
    )
    logger = logging.getLogger(__name__)
    # Set root logger to DEBUG as well
    logging.getLogger().setLevel(logging.INFO)
    return logger

# seed=31337, total_second_length=5, latent_window_size=9, steps=25, cfg=1, gs=10, rs=0, gpu_memory_preservation=6, use_teacache=True, mp4_crf=16
class GenerationRequest(BaseModel):
    prompt: str
    n_prompt: str = ""
    image_url_or_path: str
    task_id: str
    callback_url: Optional[str] = None
    seed: int = random.randint(0, sys.maxsize)
    total_second_length: float = 5.0
    latent_window_size: int = 9
    steps: int = 25
    cfg: float = 1.0
    gs: float = 10.0
    rs: float = 0.0
    gpu_memory_preservation: float = 6.0
    use_teacache: bool = True
    mp4_crf: int = 16
    
    def __str__(self):
        return f"GenerationRequest(task_id={self.task_id}, image_url_or_path={self.image_url_or_path}, prompt={self.prompt}, seed={self.seed}, total_second_length={self.total_second_length}, latent_window_size={self.latent_window_size}, steps={self.steps}, cfg={self.cfg}, gs={self.gs}, rs={self.rs}, gpu_memory_preservation={self.gpu_memory_preservation}, use_teacache={self.use_teacache}, mp4_crf={self.mp4_crf}, callback_url={self.callback_url})"
    
    def __repr__(self):
        return self.__str__()

class GenerationTask:
    def __init__(self, req: GenerationRequest, input_image: np.ndarray):
        self.req = req
        self.input_image = input_image
        self.output_filename = None
        self.is_complete = False
        # self.stream = AsyncStream()
        self.error = None
        self.progress = 0
        self.description = None
    
    def __str__(self):
        return f"GenerationTask(task_id={self.req.task_id}, image_url_or_path={self.req.image_url_or_path}, prompt={self.req.prompt}, seed={self.req.seed}, total_second_length={self.req.total_second_length}, latent_window_size={self.req.latent_window_size}, steps={self.req.steps}, cfg={self.req.cfg}, gs={self.req.gs}, rs={self.req.rs}, gpu_memory_preservation={self.req.gpu_memory_preservation}, use_teacache={self.req.use_teacache}, mp4_crf={self.req.mp4_crf}, callback_url={self.req.callback_url}, output_filename={self.output_filename}, is_complete={self.is_complete}, error={self.error}, progress={self.progress}, description={self.description})"
    
    def __repr__(self):
        return self.__str__()
    