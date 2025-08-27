<p align="center">
    <img src="https://github.com/user-attachments/assets/2cc030b4-87e1-40a0-b5bf-1b7d6b62820b" width="300">
</p>

# FramePack

Official implementation and desktop software for ["Packing Input Frame Context in Next-Frame Prediction Models for Video Generation"](https://lllyasviel.github.io/frame_pack_gitpage/).

Links: [**Paper**](https://arxiv.org/abs/2504.12626), [**Project Page**](https://lllyasviel.github.io/frame_pack_gitpage/)

FramePack is a next-frame (next-frame-section) prediction neural network structure that generates videos progressively. 

FramePack compresses input contexts to a constant length so that the generation workload is invariant to video length.

FramePack can process a very large number of frames with 13B models even on laptop GPUs.

FramePack can be trained with a much larger batch size, similar to the batch size for image diffusion training.

**Video diffusion, but feels like image diffusion.**

# News

**2025 July 14:** Some pure text2video anti-drifting stress-test results of FramePack-P1 are uploaded [here,](https://lllyasviel.github.io/frame_pack_gitpage/p1/#text-to-video-stress-tests) using common prompts without any reference images.

**2025 June 26:** Some results of FramePack-P1 are uploaded [here.](https://lllyasviel.github.io/frame_pack_gitpage/p1) The FramePack-P1 will be the next version of FramePack with two designs: Planned Anti-Drifting and History Discretization.

**2025 May 03:** The FramePack-F1 is released. [Try it here.](https://github.com/lllyasviel/FramePack/discussions/459)

Note that this GitHub repository is the only official FramePack website. We do not have any web services. All other websites are spam and fake, including but not limited to `framepack.co`, `frame_pack.co`, `framepack.net`, `frame_pack.net`, `framepack.ai`, `frame_pack.ai`, `framepack.pro`, `frame_pack.pro`, `framepack.cc`, `frame_pack.cc`,`framepackai.co`, `frame_pack_ai.co`, `framepackai.net`, `frame_pack_ai.net`, `framepackai.pro`, `frame_pack_ai.pro`, `framepackai.cc`, `frame_pack_ai.cc`, and so on. Again, they are all spam and fake. **Do not pay money or download files from any of those websites.**

# Requirements

Note that this repo is a functional desktop software with minimal standalone high-quality sampling system and memory management.

**Start with this repo before you try anything else!**

Requirements:

* Nvidia GPU in RTX 30XX, 40XX, 50XX series that supports fp16 and bf16. The GTX 10XX/20XX are not tested.
* Linux or Windows operating system.
* At least 6GB GPU memory.

To generate 1-minute video (60 seconds) at 30fps (1800 frames) using 13B model, the minimal required GPU memory is 6GB. (Yes 6 GB, not a typo. Laptop GPUs are okay.)

About speed, on my RTX 4090 desktop it generates at a speed of 2.5 seconds/frame (unoptimized) or 1.5 seconds/frame (teacache). On my laptops like 3070ti laptop or 3060 laptop, it is about 4x to 8x slower. [Troubleshoot if your speed is much slower than this.](https://github.com/lllyasviel/FramePack/issues/151#issuecomment-2817054649)

In any case, you will directly see the generated frames since it is next-frame(-section) prediction. So you will get lots of visual feedback before the entire video is generated.

# Installation

**Windows**:

[>>> Click Here to Download One-Click Package (CUDA 12.6 + Pytorch 2.6) <<<](https://github.com/lllyasviel/FramePack/releases/download/windows/framepack_cu126_torch26.7z)

After you download, you uncompress, use `update.bat` to update, and use `run.bat` to run.

Note that running `update.bat` is important, otherwise you may be using a previous version with potential bugs unfixed.

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/c49bd60d-82bd-4086-9859-88d472582b94)

Note that the models will be downloaded automatically. You will download more than 30GB from HuggingFace.

**Linux**:

We recommend having an independent Python 3.10.

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt

To start the GUI, run:

    python demo_gradio.py

Note that it supports `--share`, `--port`, `--server`, and so on.

The software supports PyTorch attention, xformers, flash-attn, sage-attention. By default, it will just use PyTorch attention. You can install those attention kernels if you know how. 

For example, to install sage-attention (linux):

    pip install sageattention==1.0.6

However, you are highly recommended to first try without sage-attention since it will influence results, though the influence is minimal.

# GUI

![ui](https://github.com/user-attachments/assets/8c5cdbb1-b80c-4b7e-ac27-83834ac24cc4)

On the left you upload an image and write a prompt.

On the right are the generated videos and latent previews.

Because this is a next-frame-section prediction model, videos will be generated longer and longer.

You will see the progress bar for each section and the latent preview for the next section.

Note that the initial progress may be slower than later diffusion as the device may need some warmup.

# Sanity Check

Before trying your own inputs, we highly recommend going through the sanity check to find out if any hardware or software went wrong. 

Next-frame-section prediction models are very sensitive to subtle differences in noise and hardware. Usually, people will get slightly different results on different devices, but the results should look overall similar. In some cases, if possible, you'll get exactly the same results.

## Image-to-5-seconds

Download this image:

<img src="https://github.com/user-attachments/assets/f3bc35cf-656a-4c9c-a83a-bbab24858b09" width="150">

Copy this prompt:

`The man dances energetically, leaping mid-air with fluid arm swings and quick footwork.`

Set like this:

(all default parameters, with teacache turned off)
![image](https://github.com/user-attachments/assets/0071fbb6-600c-4e0f-adc9-31980d540e9d)

The result will be:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/bc74f039-2b14-4260-a30b-ceacf611a185" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

**Important Note:**

Again, this is a next-frame-section prediction model. This means you will generate videos frame-by-frame or section-by-section.

**If you get a much shorter video in the UI, like a video with only 1 second, then it is totally expected.** You just need to wait. More sections will be generated to complete the video.

## Know the influence of TeaCache and Quantization

Download this image:

<img src="https://github.com/user-attachments/assets/42293e30-bdd4-456d-895c-8fedff71be04" width="150">

Copy this prompt:

`The girl dances gracefully, with clear movements, full of charm.`

Set like this:

![image](https://github.com/user-attachments/assets/4274207d-5180-4824-a552-d0d801933435)

Turn off teacache:

![image](https://github.com/user-attachments/assets/53b309fb-667b-4aa8-96a1-f129c7a09ca6)

You will get this:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/04ab527b-6da1-4726-9210-a8853dda5577" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

Now turn on teacache:

![image](https://github.com/user-attachments/assets/16ad047b-fbcc-4091-83dc-d46bea40708c)

About 30% users will get this (the other 70% will get other random results depending on their hardware):

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/149fb486-9ccc-4a48-b1f0-326253051e9b" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>A typical worse result.</em>
    </td>
  </tr>
</table>

So you can see that teacache is not really lossless and sometimes can influence the result a lot.

We recommend using teacache to try ideas and then using the full diffusion process to get high-quality results.

This recommendation also applies to sage-attention, bnb quant, gguf, etc., etc.

## Image-to-1-minute

<img src="https://github.com/user-attachments/assets/820af6ca-3c2e-4bbc-afe8-9a9be1994ff5" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/8c34fcb2-288a-44b3-a33d-9d2324e30cbd)

Set video length to 60 seconds:

![image](https://github.com/user-attachments/assets/5595a7ea-f74e-445e-ad5f-3fb5b4b21bee)

If everything is in order you will get some result like this eventually.

60s version:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/c3be4bde-2e33-4fd4-b76d-289a036d3a47" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

6s version:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/37fe2c33-cb03-41e8-acca-920ab3e34861" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

# More Examples

Many more examples are in [**Project Page**](https://lllyasviel.github.io/frame_pack_gitpage/).

Below are some more examples that you may be interested in reproducing.

---

<img src="https://github.com/user-attachments/assets/99f4d281-28ad-44f5-8700-aa7a4e5638fa" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/0e98bfca-1d91-4b1d-b30f-4236b517c35e)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/cebe178a-09ce-4b7a-8f3c-060332f4dab1" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/853f4f40-2956-472f-aa7a-fa50da03ed92" width="150">

`The girl suddenly took out a sign that said “cute” using right hand`

![image](https://github.com/user-attachments/assets/d51180e4-5537-4e25-a6c6-faecae28648a)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/116069d2-7499-4f38-ada7-8f85517d1fbb" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/6d87c53f-81b2-4108-a704-697164ae2e81" width="150">

`The girl skateboarding, repeating the endless spinning and dancing and jumping on a skateboard, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/c2cfa835-b8e6-4c28-97f8-88f42da1ffdf)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/d9e3534a-eb17-4af2-a8ed-8e692e9993d2" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/6e95d1a5-9674-4c9a-97a9-ddf704159b79" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/7412802a-ce44-4188-b1a4-cfe19f9c9118)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/e1b3279e-e30d-4d32-b55f-2fb1d37c81d2" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/90fc6d7e-8f6b-4f8c-a5df-ee5b1c8b63c9" width="150">

`The man dances flamboyantly, swinging his hips and striking bold poses with dramatic flair.`

![image](https://github.com/user-attachments/assets/1dcf10a3-9747-4e77-a269-03a9379dd9af)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/aaa4481b-7bf8-4c64-bc32-909659767115" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/62ecf987-ec0c-401d-b3c9-be9ffe84ee5b" width="150">

`The woman dances elegantly among the blossoms, spinning slowly with flowing sleeves and graceful hand movements.`

![image](https://github.com/user-attachments/assets/396f06bc-e399-4ac3-9766-8a42d4f8d383)


<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/f23f2f37-c9b8-45d5-a1be-7c87bd4b41cf" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/4f740c1a-2d2f-40a6-9613-d6fe64c428aa" width="150">

`The young man writes intensely, flipping papers and adjusting his glasses with swift, focused movements.`

![image](https://github.com/user-attachments/assets/c4513c4b-997a-429b-b092-bb275a37b719)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/62e9910e-aea6-4b2b-9333-2e727bccfc64" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

# Prompting Guideline

Many people would ask how to write better prompts. 

Below is a ChatGPT template that I personally often use to get prompts:

    You are an assistant that writes short, motion-focused prompts for animating images.

    When the user sends an image, respond with a single, concise prompt describing visual motion (such as human activity, moving objects, or camera movements). Focus only on how the scene could come alive and become dynamic using brief phrases.

    Larger and more dynamic motions (like dancing, jumping, running, etc.) are preferred over smaller or more subtle ones (like standing still, sitting, etc.).

    Describe subject, then motion, then other things. For example: "The girl dances gracefully, with clear movements, full of charm."

    If there is something that can dance (like a man, girl, robot, etc.), then prefer to describe it as dancing.

    Stay in a loop: one image in, one motion prompt out. Do not explain, ask questions, or generate multiple options.

You paste the instruct to ChatGPT and then feed it an image to get prompt like this:

![image](https://github.com/user-attachments/assets/586c53b9-0b8c-4c94-b1d3-d7e7c1a705c3)

*The man dances powerfully, striking sharp poses and gliding smoothly across the reflective floor.*

Usually this will give you a prompt that works well. 

You can also write prompts yourself. Concise prompts are usually preferred, for example:

*The girl dances gracefully, with clear movements, full of charm.*

*The man dances powerfully, with clear movements, full of energy.*

and so on.

# Cite

    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }
