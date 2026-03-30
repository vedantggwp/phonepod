# System Architecture: Project Resonance

## Tech Stack
* **Python Manager:** `uv`
* **Audio I/O:** `torchaudio`
* **AI Model:** `resemble-enhance` (from GitHub: `https://github.com/resemble-ai/resemble-enhance.git`)
* **UI:** `gradio`

## Module Definitions (Strict Separation)

### 1. `engine.py` (The AI Core)
* **Purpose:** Wraps the `resemble-enhance` model.
* **Input:** A `torch.Tensor` (16kHz, mono) and the sample rate.
* **Process:** Initializes the dual-stage pipeline (UNet Denoiser + Continuous Flow Matching). Uses `nfe=64` and `solver="midpoint"`.
* **Output:** A high-fidelity `torch.Tensor`.
* **Constraint:** This module handles NO file I/O. It only processes tensors in memory.

### 2. `processor.py` (The Chunking Manager)
* **Purpose:** Solves the Out-Of-Memory (OOM) crash hazard for long audio files.
* **Input:** A file path to a degraded `.wav` file.
* **Process:**
    * Loads the audio via `torchaudio`.
    * Implements an **Overlap-Add (OLA)** chunking strategy.
    * Slices audio into 10-second chunks with a 1-second overlap.
    * Passes each chunk to `engine.py`.
    * Reconstructs the audio using a Hanning window crossfade across the overlaps to prevent audio clicking.
* **Output:** Saves the final high-fidelity `.wav` to disk.

### 3. `app.py` (The Interface)
* **Purpose:** The user-facing drag-and-drop web UI.
* **Process:** Uses Gradio to accept an audio file, passes the path to `processor.py`, and displays an A/B comparison player upon completion.
