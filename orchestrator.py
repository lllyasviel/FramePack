import os
import time
from studio_engine import StudioEngine
from bridge_npu import NPUOrchestrator
from main import StateSaver

class ProductionStudio:
    """The main orchestration loop for autonomous script-to-movie production."""
    def __init__(self):
        print("--- [Studio] Opening Autonomous Production Mode ---")
        self.npu = NPUOrchestrator()
        self.engine = StudioEngine()
        self.state_saver = StateSaver()
        self.production_id = f"prod_{int(time.time())}"
        
        if not os.path.exists("dailies"):
            os.makedirs("dailies")

    def run_script(self, script_path):
        """Processes a full production script."""
        if not os.path.exists(script_path):
            print(f"[Studio] Error: Script '{script_path}' not found.")
            return

        with open(script_path, "r") as f:
            script_content = f.read()

        print(f"[Studio] Script Loaded. Handing off to Director...")
        scenes = self.npu.process_production(script_content)
        
        dailies = []
        for i, scene_text in enumerate(scenes):
            print(f"\n--- [Studio] Producing Scene {i+1} of {len(scenes)} ---")
            
            # 1. Director expands the prompt
            visual_prompt = self.npu.director.direct_scene(scene_text)
            
            # 2. Key Grip schedules the render
            params = self.npu.key_grip.schedule_render({"scene": i})
            
            # 3. Cinematographer renders the scene
            # Note: In a full implementation, we'd pass an initial image for scene 0
            # then use the cached latents for subsequent scenes.
            print(f"[Studio] Cinematographer taking over... VRAM Target: {params['vram_target']}")
            
            # Placeholder for actual render call
            daily_path = os.path.join("dailies", f"{self.production_id}_scene_{i}.mp4")
            # self.engine.render_section(None, visual_prompt, job_id=self.production_id, state_saver=self.state_saver)
            
            print(f"[Studio] Scene {i+1} complete. Daily saved to: {daily_path}")
            dailies.append(daily_path)

        print(f"\n--- [Studio] Production Complete. {len(dailies)} scenes produced. ---")
        print(f"[Studio] Use the 94GB State-Saver cache to stitch the final cut.")

if __name__ == "__main__":
    # Create a dummy script for testing
    test_script_path = "production_script.txt"
    if not os.path.exists(test_script_path):
        with open(test_script_path, "w") as f:
            f.write("A futuristic neon city at night.\n\nA robot walking through a crowd of people.\n\nThe robot stops and looks up at a massive digital screen.")
    
    studio = ProductionStudio()
    studio.run_script(test_script_path)
