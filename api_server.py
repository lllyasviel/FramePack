import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from scheduler import Scheduler
from utils import GenerationRequest, GenerationTask, setup_logging

logger = setup_logging()

class APIServer:
    def __init__(self, args, inference):
        # Only initialize API server on rank 0
        if inference.rank != 0:
            return
            
        self.app = FastAPI(title="FramePack API")
        self.args = args
        self.inference = inference
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize scheduler
        self.scheduler = Scheduler(args, inference)
            
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        @self.app.post("/generate")
        async def generate_video(req: GenerationRequest, request: Request):
            if req.task_id in self.scheduler.active_tasks:
                raise HTTPException(status_code=400, detail=f"Task ID already exists, active_tasks: {self.scheduler.active_tasks}, req.task_id: {req.task_id}")
            
            # Add task to queue
            self.scheduler.task_queue.put(GenerationTask(req=req, input_image=self.scheduler.get_image(req.image_url_or_path)))
            logger.info(f"Task queued. Queue size: {self.scheduler.task_queue.qsize()}")
            
            return {"task_id": req.task_id, "status": "queued"}

        @self.app.get("/status/{task_id}")
        async def get_status(task_id: str, request: Request):
            if task_id not in self.scheduler.active_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.scheduler.active_tasks[task_id]
            
            if task.error:
                return {
                    "task_id": task_id,
                    "status": "error",
                    "error": task.error,
                    "description": task.description
                }
            
            if task.is_complete:
                return {
                    "task_id": task_id,
                    "status": "complete",
                    "video_path": task.output_filename,
                    "progress": 100
                }
            
            return {
                "task_id": task_id,
                "status": "processing",
                "progress": task.progress,
                "description": task.description
            }

        @self.app.get("/video/{task_id}")
        async def get_video(task_id: str, request: Request):
            if task_id not in self.scheduler.active_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.scheduler.active_tasks[task_id]
            
            if not task.is_complete or not task.output_filename:
                raise HTTPException(status_code=400, detail="Video not ready")
            
            def iterfile():
                with open(task.output_filename, "rb") as f:
                    while chunk := f.read(1024*1024):
                        yield chunk
            
            return StreamingResponse(
                iterfile(),
                media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="framepack_{task_id}.mp4"'}
            )
    
    def run(self):
        """Run the FastAPI server"""
        # Only run on rank 0
        if self.inference.rank != 0:
            return
            
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.args.port,
            reload=False
        ) 