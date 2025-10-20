import os
import pickle
import tempfile
import time
import traceback
from typing import Any, Dict, Optional
import argparse

# Third-party libraries
import einops
import numpy as np
import torch
import torch.distributed as dist
import xfuser
import zmq
from diffusers import AutoencoderKLHunyuanVideo
from PIL import Image
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    LlamaModel,
    LlamaTokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

# Local libraries
from api_server import APIServer
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_decode_fake,
    vae_decode_parallel,
    vae_encode,
)
from diffusers_helper.memory import (
    DynamicSwapInstaller,
    cpu,
    fake_diffusers_current_device,
    get_cuda_free_memory_gb,
    load_model_as_complete,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    unload_complete_models,
)
from diffusers_helper.models.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked, HunyuanVideoTransformerBlock, HunyuanVideoSingleTransformerBlock
)
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.utils import (
    crop_or_pad_yield_mask,
    generate_timestamp,
    resize_and_center_crop,
    save_bcthw_as_mp4,
    soft_append_bcthw,
)
from utils import setup_logging

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=8081, required=False)
parser.add_argument("--enable_fuse_qkv", action='store_true', default=False)
parser.add_argument("--ulysses_degree", type=int, default=1)
parser.add_argument("--ring_degree", type=int, default=1)
parser.add_argument("--torch_compile_mode", type=str, default=None, choices=["max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead"])
args = parser.parse_args()

# Initialize logger with rank 0 (will be updated in distributed mode)
logger = setup_logging()

logger.info(args)


class FramePackI2VInference:
    def __init__(
        self,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        enable_fuse_qkv: bool = False,
        device: Optional[str] = None,
        args = None
    ):
        """
        Initialize FramePackI2V inference with Ulysses parallel support.
        
        Args:
            ulysses_degree: Degree of Ulysses parallelization
            ring_degree: Degree of ring parallelization
            device: Device to run inference on
            args: Command line arguments for server configuration
        """
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.enable_fuse_qkv = enable_fuse_qkv
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        
        self.rank = 0
        self.world_size = 1
        self.torch_compile_mode = args.torch_compile_mode
        
        # Initialize distributed environment
        self._init_distributed()
        
        # Initialize ZMQ context and sockets
        self._init_zmq()
        
        # Initialize outputs folder
        self.outputs_folder = './outputs/'
        os.makedirs(self.outputs_folder, exist_ok=True)

        # Load models
        self._load_models()
    
    def _init_zmq(self):
        """Initialize ZMQ context and sockets for task distribution"""
        try:
            # Create ZMQ context
            self.zmq_context = zmq.Context()
            logger.debug(f"Rank {self.rank} created ZMQ context")
            
            # Use fixed path for socket file
            socket_path = "/tmp/framepack_task.ipc"
            logger.debug(f"Rank {self.rank} using socket path: {socket_path}")
            
            # Initialize socket based on rank
            if self.rank == 0:
                # Rank 0 uses PUB socket to broadcast tasks
                self.task_socket = self.zmq_context.socket(zmq.PUB)
                # Remove existing socket file if it exists
                if os.path.exists(socket_path):
                    os.remove(socket_path)
                self.task_socket.bind(f"ipc://{socket_path}")
                logger.info(f"Rank {self.rank} bound to socket: {socket_path}")
                logger.debug(f"Rank {self.rank} PUB socket created and bound")
                
                # Wait for other ranks to connect
                time.sleep(2)  # Give time for other ranks to connect
                logger.debug(f"Rank {self.rank} finished waiting for connections")
            else:
                # Other ranks use SUB socket to receive tasks
                self.task_socket = self.zmq_context.socket(zmq.SUB)
                self.task_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
                logger.debug(f"Rank {self.rank} SUB socket created and subscribed to all messages")
                
                self.task_socket.connect(f"ipc://{socket_path}")
                logger.info(f"Rank {self.rank} connected to socket: {socket_path}")
                logger.debug(f"Rank {self.rank} SUB socket connected")
                
                # Set socket options for better reliability
                self.task_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
                logger.debug(f"Rank {self.rank} set socket timeout to 1000ms")
                
        except Exception as e:
            logger.error(f"Rank {self.rank} Error initializing ZMQ: {e}")
            logger.error(traceback.format_exc())
            raise

    def run(self):
        # Synchronize all ranks before starting server or waiting for tasks
        if self.world_size > 1:
            dist.barrier()
        
        # Initialize API server on rank 0
        if self.rank == 0 and self.args is not None:
            self.api_server = APIServer(self.args, self)
            self.api_server.run()
        else:
            # Non-rank 0 processes wait for tasks
            self._wait_for_tasks()
    
    def _wait_for_tasks(self):
        """Wait for and process tasks from rank 0"""
        logger.info(f"Rank {self.rank} starting task wait loop")
        while True:
            try:
                # Wait for task from rank 0
                task = self._receive_task()
                if task is not None:
                    logger.info(f"received task: {task.req.task_id}")
                    self._process_task(task)
                    
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)  # Sleep before retrying

    def _receive_task(self):
        """Receive task from rank 0 using ZMQ"""
        if self.world_size <= 1:
            return None

        try:
            # Try to receive task with timeout
            task_data = self.task_socket.recv(flags=zmq.NOBLOCK)
            if task_data:
                task = pickle.loads(task_data)
                return task
        except zmq.Again:
            pass
        except Exception as e:
            logger.error(f"Rank {self.rank} Error receiving task: {e}")
            logger.error(traceback.format_exc())
            
        return None

    def _broadcast_task(self, task):
        """Broadcast task to all ranks using ZMQ"""
        if self.world_size <= 1:
            return
            
        assert self.rank == 0, "Only rank 0 can broadcast task"
        try:
            # Serialize and broadcast task
            task_data = pickle.dumps(task)
            self.task_socket.send(task_data)
        except Exception as e:
            logger.error(f"Error broadcasting task: {e}")
            logger.error(traceback.format_exc())
            raise

    def _init_distributed(self):
        """Initialize distributed environment"""
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        logger.info(f"{os.environ}")
        assert world_size == self.ring_degree * self.ulysses_degree, \
            f"World size {world_size} must equal ring_degree {self.ring_degree} * ulysses_degree {self.ulysses_degree}"
        init_distributed_environment(rank=rank, world_size=world_size)
        initialize_model_parallel(
            sequence_parallel_degree=world_size,
            ring_degree=self.ring_degree,
            ulysses_degree=self.ulysses_degree,
        )
        # Get local rank from environment variable
        self.rank = get_sequence_parallel_rank()
        self.world_size = get_sequence_parallel_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

        setup_logging(self.rank)
        
        logger.info(f"Initialized distributed environment - rank {self.rank} of {self.world_size} on device {self.device}")
            
    def _load_models(self):
        """Load all required models"""
        self.model_paths = {
            'hunyuanvideo': "/mnt/llm_data/models/hunyuanvideo-community/HunyuanVideo",
            'flux_redux_bfl': "/mnt/llm_data/models/lllyasviel/flux_redux_bfl",
            'framepack_i2v_hy': "/mnt/llm_data/models/lllyasviel/FramePackI2V_HY"
        }
        # Load text encoders
        self.text_encoder = LlamaModel.from_pretrained(
            self.model_paths['hunyuanvideo'],
            subfolder='text_encoder',
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cpu()
        
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            self.model_paths['hunyuanvideo'],
            subfolder='text_encoder_2',
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cpu()
        
        # Load tokenizers
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            self.model_paths['hunyuanvideo'],
            subfolder='tokenizer',
            trust_remote_code=True
        )
        
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.model_paths['hunyuanvideo'],
            subfolder='tokenizer_2',
            trust_remote_code=True
        )
        
        # Load VAE
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            self.model_paths['hunyuanvideo'],
            subfolder='vae',
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        
        # Load image encoder and feature extractor
        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            self.model_paths['flux_redux_bfl'],
            subfolder='feature_extractor',
            trust_remote_code=True
        )
        
        self.image_encoder = SiglipVisionModel.from_pretrained(
            self.model_paths['flux_redux_bfl'],
            subfolder='image_encoder',
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cpu()
        
        # Load transformer
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            self.model_paths['framepack_i2v_hy'],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cpu()

        # enable fuse qkv
        if self.enable_fuse_qkv:
            self.fuse_model_attn_projections(self.transformer)
            logger.info(f'enable fuse qkv')

        # Set models to eval mode
        for model in [self.vae, self.text_encoder, self.text_encoder_2, 
                     self.image_encoder, self.transformer]:
            model.eval() 
            model.requires_grad_(False)

        # Check available VRAM and set high_vram mode
        free_mem_gb = get_cuda_free_memory_gb(self.device)
        self.high_vram = free_mem_gb > 60
        logger.info(f'Free VRAM {free_mem_gb} GB')
        logger.info(f'High-VRAM Mode: {self.high_vram}')

        # Configure models based on VRAM availability
        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
            DynamicSwapInstaller.install_model(self.transformer, device=self.device)
            DynamicSwapInstaller.install_model(self.text_encoder, device=self.device)
        else:
            self.text_encoder.to(self.device)
            self.text_encoder_2.to(self.device)
            self.image_encoder.to(self.device)
            self.vae.to(self.device)
            self.transformer.to(self.device)

        logger.info(f"torch_compile_mode: {self.torch_compile_mode}")
        if self.torch_compile_mode == 'max-autotune-no-cudagraphs':
            logger.info('Using torch compile: max-autotune-no-cudagraphs')
            torch._dynamo.config.capture_scalar_outputs = True
            self.transformer = torch.compile(self.transformer, mode="max-autotune-no-cudagraphs")
        elif self.torch_compile_mode == None:
            logger.info('Not using torch compile')
        else:
            assert False, f"Invalid torch compile mode: {self.torch_compile_mode}"

        self.transformer.high_quality_fp32_output_for_inference = True
        logger.info('transformer.high_quality_fp32_output_for_inference = True')

    def fuse_model_attn_projections(self, model):
        for module in model.modules():
            if isinstance(module, HunyuanVideoTransformerBlock) or isinstance(module, HunyuanVideoSingleTransformerBlock):
                if hasattr(module.attn, 'fuse_projections'):
                    module.attn.fuse_projections()

    def enqueue_task(self, task):
        """Enqueue a generation task with distributed support"""
        try:
            self.current_task = task
            if self.rank == 0:
                self._broadcast_task(task)
            result = self._process_task(task)
            if self.rank == 0:
                # task.stream.output_queue.push(('end', None))
                task.is_complete = True
                
            return result
            
        except Exception as e:
            logger.error(f"rank {self.rank} Error in worker: {e}")
            logger.error(traceback.format_exc())
            if self.rank == 0:
                task.error = True
                task.description = f"Error: {e}\n{traceback.format_exc()}"
                task.is_complete = True
            raise

    def _process_task(self,task):
        with torch.inference_mode():
            self.inter_process_task(task)

    @torch.no_grad()
    def inter_process_task(self, task):
        task_id = task.req.task_id
        input_image = task.input_image
        prompt = task.req.prompt
        n_prompt = task.req.n_prompt
        seed = task.req.seed
        total_second_length = task.req.total_second_length
        latent_window_size = task.req.latent_window_size
        steps = task.req.steps
        cfg = task.req.cfg
        gs = task.req.gs
        rs = task.req.rs
        gpu_memory_preservation = task.req.gpu_memory_preservation
        use_teacache = task.req.use_teacache
        mp4_crf = task.req.mp4_crf
        
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        job_id = generate_timestamp()

        logger.info(f"Rank {self.rank} start task {task_id}")
        start_time = time.time()
        try:
            # Clean GPU
            if not self.high_vram:
                unload_complete_models(
                    self.text_encoder, self.text_encoder_2, self.image_encoder, self.transformer
                )

            # Text encoding
            if not self.high_vram:
                fake_diffusers_current_device(self.text_encoder, self.device)
                load_model_as_complete(self.text_encoder_2, target_device=self.device)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
            logger.debug(f"Rank {self.rank} llama_vec device: {llama_vec.device}, clip_l_pooler device: {clip_l_pooler.device}")

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
            logger.debug(f"Rank {self.rank} llama_vec_n device: {llama_vec_n.device}, clip_l_pooler_n device: {clip_l_pooler_n.device}")

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            logger.debug(f"Rank {self.rank} llama_vec device after crop/pad: {llama_vec.device}, llama_attention_mask device: {llama_attention_mask.device}")

            # Processing input image
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            Image.fromarray(input_image_np).save(os.path.join(self.outputs_folder, f'{job_id}.png'))

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
            logger.debug(f"Rank {self.rank} input_image_pt device: {input_image_pt.device}")

            # VAE encoding
            if not self.high_vram:
                load_model_as_complete(self.vae, target_device=self.device)

            start_latent = vae_encode(input_image_pt, self.vae)
            logger.debug(f"Rank {self.rank} start_latent device: {start_latent.device}")

            # CLIP Vision
            if not self.high_vram:
                load_model_as_complete(self.image_encoder, target_device=self.device)

            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            logger.debug(f"Rank {self.rank} image_encoder_last_hidden_state device: {image_encoder_last_hidden_state.device}")

            # Dtype
            llama_vec = llama_vec.to(self.transformer.dtype)
            llama_vec_n = llama_vec_n.to(self.transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)
            logger.debug(f"Rank {self.rank} After dtype conversion - llama_vec device: {llama_vec.device}, image_encoder_last_hidden_state device: {image_encoder_last_hidden_state.device}")

            # Move tensors to correct device
            llama_vec = llama_vec.to(self.device)
            llama_vec_n = llama_vec_n.to(self.device)
            clip_l_pooler = clip_l_pooler.to(self.device)
            clip_l_pooler_n = clip_l_pooler_n.to(self.device)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.device)
            llama_attention_mask = llama_attention_mask.to(self.device)
            llama_attention_mask_n = llama_attention_mask_n.to(self.device)
            logger.debug(f"Rank {self.rank} After device move - llama_vec device: {llama_vec.device}, image_encoder_last_hidden_state device: {image_encoder_last_hidden_state.device}")

            # Sampling
            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            latent_paddings = reversed(range(total_latent_sections))

            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            for latent_padding in latent_paddings:
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                logger.debug(f"Rank {self.rank} clean_latents device: {clean_latents.device}")
                logger.debug(f"Rank {self.rank} clean_latents_2x device: {clean_latents_2x.device}")
                logger.debug(f"Rank {self.rank} clean_latents_4x device: {clean_latents_4x.device}")
                
                clean_latents = clean_latents.to(self.device)
                clean_latents_2x = clean_latents_2x.to(self.device)
                clean_latents_4x = clean_latents_4x.to(self.device)

                if not self.high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(self.transformer, target_device=self.device, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    self.transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                    return

                logger.debug(f"Rank {self.rank} Before sample_hunyuan - transformer device: {next(self.transformer.parameters()).device}")
                generated_latents = sample_hunyuan(
                    transformer=self.transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=self.device,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )
                logger.debug(f"Rank {self.rank} After sample_hunyuan - generated_latents device: {generated_latents.device}")

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
                logger.debug(f"Rank {self.rank} history_latents device: {history_latents.device}")

                if not self.high_vram:
                    offload_model_from_device_for_memory_preservation(self.transformer, target_device=self.device, preserved_memory_gb=8)
                    load_model_as_complete(self.vae, target_device=self.device)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode_parallel(real_history_latents, self.vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_pixels = vae_decode_parallel(real_history_latents[:, :, :section_latent_frames], self.vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                if not self.high_vram:
                    unload_complete_models()
                if get_sequence_parallel_rank() == 0:
                    output_filename = os.path.join(self.outputs_folder, f'{task_id}_{job_id}_{total_generated_latent_frames}_{self.rank}.mp4')
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

                    if is_last_section:
                        task.output_filename = output_filename
                        logger.info(f'last_section, Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                if is_last_section:
                    break
        except Exception as e:
            logger.error(traceback.format_exc())
            task.error = True
            task.description = f"Error: {e}\n{traceback.format_exc()}"
            if not self.high_vram:
                unload_complete_models(
                    self.text_encoder, self.text_encoder_2, self.image_encoder, self.transformer
                )
        finally:
            task.is_complete = True
        end_time = time.time()
        logger.info(f"Rank {self.rank} end task {task_id}, cost time: {end_time - start_time:.2f} seconds")

    def __del__(self):
        """Cleanup ZMQ resources and temporary files"""
        if hasattr(self, 'task_socket'):
            self.task_socket.close()
        if hasattr(self, 'zmq_context'):
            self.zmq_context.term()
            
        # Clean up temporary directory if we're rank 0
        if hasattr(self, 'temp_dir') and self.rank == 0:
            try:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")

if __name__ == "__main__":
    # Initialize inference with Ulysses parallel
    inference = FramePackI2VInference(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        enable_fuse_qkv=args.enable_fuse_qkv,
        args=args
    )
    inference.run()