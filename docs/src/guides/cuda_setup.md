# CUDA Setup Guide

Shrew's CUDA backend requires NVIDIA libraries to run. You have two options: a **System-wide Installation** (recommended if you are a developer) or a **Local/Portable Setup** (useful for distributing the app or isolating versions).

## Option A: System-wide Installation (Recommended)

If you have valid NVIDIA drivers and the **CUDA Toolkit** installed, you generally **do not need** to do anything else.

1.  **Check Driver**: Ensure your NVIDIA driver is up to date (supports CUDA 12+).
2.  **Check Toolkit**: Ensure `nvcc --version` works in your terminal.
3.  **Run**:
    ```powershell
    cargo run --release -p example-gpu-demo --features cuda
    ```

*Note: If Shrew complains about missing DLLs (e.g., `nvrtc64_120.dll`), your Toolkit version might be too old or not in your PATH.*

## Option B: Local/Portable Setup (`cuda_libs`)

Use this method if you don't want to install the full CUDA Toolkit or need to ensure a specific CUDA version matches the binary.

### 1. Prepare `cuda_libs`

We have created a `cuda_libs/` folder in the root of the repository. You must populate this folder with the necessary CUDA dynamic libraries (DLLs).
You can run the following PowerShell commands to automatically download them from PyPI:

```powershell
# 1. Create temp folder and download packages (requires Python installed)
mkdir temp_dl -Force
python -m pip download --no-deps -d temp_dl nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12

# 2. Extract and move DLLs
Get-ChildItem temp_dl/*.whl | ForEach-Object { 
    $extractPath = "temp_dl/extract_" + $_.BaseName
    Expand-Archive -Path $_.FullName -DestinationPath $extractPath
    Get-ChildItem -Path $extractPath -Recurse -Filter "*.dll" | Move-Item -Destination "cuda_libs/" -Force
}

# 3. Clean up
Remove-Item temp_dl -Recurse -Force
```

**Required Files (CUDA 12.x):**
- `nvrtc64_120.dll` (CUDA 12.0+)
- `nvrtc-builtins64_120.dll`
- `cudart64_12.dll`
- `cublas64_12.dll`
- `cublasLt64_12.dll`

### 2. Update System PATH

Since you moved the repository, your previous environment configuration is likely broken. To fix this permanently without using scripts:

1.  Press `Win + R`, type `sysdm.cpl`, and hit Enter.
2.  Go to the **Advanced** tab -> **Environment Variables**.
3.  Under **User variables for [YourUser]**, find `Path` (or create it if it doesn't exist) and click **Edit**.
4.  Click **New** and paste the full path to your `cuda_libs` folder:
    ```
    C:\Users\Juan Simancas\Documents\TheShrewFoundation\shrew\cuda_libs
    ```
5.  Click **OK** on all dialogs.

## Verification

Close your current terminal and open a new one (to load the new PATH).

Run the GPU demo:
```powershell
cargo run --release -p example-gpu-demo --features cuda
```

If successful, you should see GPU initialization logs and matrix multiplication benchmarks.
