import gc
import os
import pathlib
import subprocess
import threading
import time
import traceback
from pathlib import Path
from queue import Queue
from threading import Lock

import cv2
import numpy as np
import requests
import torch
from cachetools import LRUCache
from fastapi import HTTPException

from utils import GenerationTask, setup_logging

logger = setup_logging()

class Scheduler:
    def __init__(self, args, inference):
        # Only initialize scheduler on rank 0
        if inference.rank != 0:
            return
            
        self.args = args
        self.inference = inference
        self.rank = inference.rank
        self.task_queue = Queue()
        self.active_tasks = LRUCache(maxsize=30)
        self.processing_lock = Lock()

        # Start task processor thread
        self.processor_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.processor_thread.start()
    
    def get_image(self, image_source: str) -> np.ndarray:
        """
        Get image from local path or URL
        Args:
            image_source: Local path or URL to image
        Returns:
            numpy array of image in RGB format
        """
        try:
            # Check if it's a URL
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source)
                response.raise_for_status()
                nparr = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Local file
                if not Path(image_source).exists():
                    raise FileNotFoundError(f"Image file not found: {image_source}")
                img = cv2.imread(image_source)
                if img is None:
                    raise ValueError(f"Failed to read image: {image_source}")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
            
    def get_webp_video_path(self, local_video_path: str, webp_video_path: str, timeout: int = 300):
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(webp_video_path)
        os.makedirs(output_dir, exist_ok=True)

        # Build ffmpeg command for webp conversion
        cmd = [
            'ffmpeg',
            '-i', local_video_path,
            '-vf', 'fps=8',
            '-s', '270x454',
            '-quality', '90',
            '-loop', '0',
            webp_video_path
        ]

        try:
            # Run ffmpeg command and wait for completion with timeout
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
            
            # Check return code
            if result.returncode != 0:
                logger.error(f'Error converting video to webp: {result.stderr}')
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            return webp_video_path
        except subprocess.TimeoutExpired as e:
            logger.error(f'Conversion timed out after {timeout} seconds')
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f'Error converting video to webp: {e.stderr}')
            raise
    
    def _process_tasks(self):
        """Process tasks from queue in a separate thread"""
        # Only run on rank 0
        if self.rank != 0:
            return
            
        logger.info("Task processor thread started")
        while True:
            try:
                # Sleep first to ensure proper timing between attempts
                time.sleep(1)

                if self.task_queue.empty():
                    continue

                # Acquire lock for task processing
                if self.processing_lock.acquire(blocking=False):
                    try:
                        # Process task if available
                        try:
                            task = self.task_queue.get()
                            logger.info(f"Processing task: {task}")
                            self.active_tasks[task.req.task_id] = task
                            self.process_task(task)
                            logger.info(f"{task.output_filename=}")
                            self.task_queue.task_done()
                        finally:
                            gc.collect()
                            torch.cuda.empty_cache()
                    finally:
                        self.processing_lock.release()
                
            except Exception as e:
                logger.error(f"Error in task processor: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(1)

    def process_task(self, task):
        try:
            self.inference.enqueue_task(task)
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            task.error = True
            task.description = f"Error: {e}\n{traceback.format_exc()}"
        finally:
            task.is_complete = True
