import time
from datetime import datetime

import requests

from utils import GenerationRequest


def generate_video(image_url_or_path: str, task_id: str, prompt: str, **kwargs):
    """
    Generate video using FramePack API
    Args:
        image_url_or_path: URL or local path of the image
        task_id: Unique task identifier
        **kwargs: Other parameters for video generation
    """
    # Prepare the request data with image_url_or_path and task_id
    data = {
        "image_url_or_path": image_url_or_path,
        "task_id": task_id,
        "prompt": prompt,
        **kwargs  # Unpack all other parameters directly
    }

    print(f"Request data: {data}")

    # Start generation
    response = requests.post(
        "http://localhost:8098/generate",
        json=data
    )

    if response.status_code != 200:
        print(f"Error response: {response.text}")

    response.raise_for_status()
    result = response.json()
    print(f"Result: {result}")
    task_id = result["task_id"]
    time.sleep(5)  # Increased sleep time to ensure task is picked up
    
    # Poll for status
    while True:
        status = requests.get(f"http://localhost:8098/status/{task_id}")
        status_data = status.json()
        print(f"Status: {status_data}")
        
        if status_data["status"] == "complete":
            break
        elif status_data["status"] == "error":
            raise Exception(f"Generation failed: {status_data.get('error')}")
        
        time.sleep(15)

    # Download video
    video = requests.get(f"http://localhost:8081/video/{task_id}")
    output_path = f"./outputs/output_{task_id}.mp4"
    with open(output_path, "wb") as f:
        f.write(video.content)
    
    return output_path

# Example usage
if __name__ == "__main__":
    image_url = "test.jpg"
    
    output_file = generate_video(
        image_url_or_path=image_url,
        task_id=f"test_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        prompt="The girl dances gracefully",
        total_second_length=5.0,
        latent_window_size=9,
        steps=25,
        cfg=1.0,
        gs=10.0,
        rs=0.0,
        gpu_memory_preservation=6.0,
        use_teacache=True,
        mp4_crf=16
    )
    print(f"Video saved to: {output_file}")
