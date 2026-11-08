# AMD DIRECTOR STUDIO - Final Cut

**Repository:** [Cyb3rLab5/amd-strix-point](https://github.com/Cyb3rLab5/amd-strix-point)

I have successfully forged a native AMD-optimized movie studio using the FramePack architecture, specifically tailored for your **Ryzen 9 AI beast**.

## 🎭 The Director Studio Experience

The interface has been upgraded to a premium **"Director Studio"** theme with glassmorphism and real-time monitoring.

### 🧩 One-Click Launch (Pinokio)
I've added support for **Pinokio** (optional). If you choose to install it later:
- **`pinokio.js`**: Automatically recognizes the project.
- **`install.json`**: Handles the AMD environment setup.
- **`start.json`**: Launches the studio and opens your browser.

### 🎨 Premium UI Features
- **Cinematic Design**: Glassmorphism dark theme optimized for creative focus.
- **NPU Status Monitor**: Live tracking of your **Director** and **Key Grip** agents on the XDNA 2 NPU.
- **64GB VRAM Leverage**: Native access to the expanded context packing buffer (Default `latent_window_size` = 17).

## 🚀 How to Launch (Manual)

1.  **Open Terminal** in the project folder.
2.  **Launch Studio**:
    ```powershell
    .\.venv\Scripts\python.exe demo_gradio.py
    ```

## 💎 AMD Optimizations Included

- **Radeon 890M iGPU**: All diffusion accelerated via **DirectML**.
- **XDNA 2 NPU**: `bridge_npu.py` handles the logic roles (Director/Key Grip).
- **94GB RAM State-Saver**: Caches latent history to prevent video drift.

---
*Optimized for AMD Strix Point by Antigravity AI*
