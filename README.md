# FramePack - AMD Strix Point Edition 🎬🚀

This is a specialized fork of [lllyasviel/FramePack](https://github.com/lllyasviel/FramePack), natively optimized for **AMD Ryzen 9 AI (Strix Point)** hardware. 

**Official AMD Fork:** [Cyb3rLab5/amd-strix-point](https://github.com/Cyb3rLab5/amd-strix-point)

---

## 💎 Exclusive AMD Enhancements

### 1. Native DirectML Acceleration (No CUDA required)
The entire pixel diffusion pipeline has been patched to run on **DirectML** via `torch-directml`. This allows native execution on the **Radeon 890M** without any Nvidia dependencies or overhead.

### 2. High-RAM "State-Saver" Logic
Utilizes up to **94GB of System RAM** to cache the entire latent history of a production. This prevents "visual drift" in long-sequence renders (60s+), ensuring narrative and temporal coherence that standard VRAM-only systems cannot maintain.

### 3. Dual-Agent NPU Orchestration
Offloads logic-heavy, non-pixel tasks to the **XDNA 2 NPU** via `bridge_npu.py`:
- **The Director (NPU)**: Handles prompt expansion and maintaining stylistic consistency.
- **The Key Grip (NPU)**: Manages frame scheduling, latent caching logistics, and VRAM floor monitoring.

### 4. Optimized Context Packing
Leverages the **64GB UMA VRAM** allocation. The default `latent_window_size` is increased from 9 to **17**, allowing for significantly higher contextual awareness and smoother video chunking.

---

## 🎨 Premium "Director Studio" UI

This fork includes a revamped Gradio interface featuring:
- **Glassmorphism Design**: A cinematic dark theme for creative focus.
- **NPU Status Monitor**: Real-time tracking of the Director and Key Grip agents.
- **Live Latent Preview**: Visual monitoring of the diffusion progress.

---

## 🛠 Installation & Launch

### Manual Setup
1. Clone this fork.
2. Run the specialized AMD installer:
   ```powershell
   python setup_amd.py
   ```
3. Launch the Studio:
   ```powershell
   .\.venv\Scripts\python.exe demo_gradio.py
   ```

### Pinokio Setup (One-Click)
This repo includes `pinokio.js`, `install.json`, and `start.json`. Simply open this folder in **Pinokio** to automate the entire environment setup and launch process.

---

## 🤖 Autonomous Production (Headless)

For "Script-to-Movie" workflows, use the autonomous orchestrator:
1. Edit `production_script.txt` with your scene descriptions.
2. Run the orchestrator:
   ```powershell
   .\.venv\Scripts\python.exe orchestrator.py
   ```
The studio will then run the production end-to-end, saving "dailies" to the `dailies/` folder without user intervention.

---

## 📊 Verification
Run the built-in benchmark to verify your Strix Point configuration:
```powershell
.\.venv\Scripts\python.exe verify_strix.py
```

---

*Fork maintained and optimized for AMD Ryzen AI by Antigravity AI.*
