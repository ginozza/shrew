# Security Policy

## Supported Versions

The following versions of Shrew are currently supported with security updates:

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a Vulnerability

I take the security of Shrew seriously. If you discover a security vulnerability, I appreciate your responsible disclosure.

### How to Report

1. **Do NOT open a public issue.** Security vulnerabilities should not be reported through public GitHub issues.
2. **Use GitHub Security Advisories:** Open a [draft security advisory](https://github.com/ginozza/shrew/security/advisories/new) in the repository.
3. **Email:** You can also reach me directly at jsimancas@unimagdalena.edu.co.

### What to Include

When reporting a vulnerability, please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce the issue or a proof-of-concept.
- The affected crate(s) (e.g., `shrew-core`, `shrew-cuda`, `shrew-ir`, `shrew-python`).
- The affected version(s).
- Any suggested fix or mitigation, if available.

### Response Timeline

- **Acknowledgment:** I will acknowledge receipt of your report within **48 hours**.
- **Assessment:** I will provide an initial assessment within **7 days**.
- **Resolution:** I aim to release a fix within **30 days** for confirmed vulnerabilities, depending on complexity.

### What to Expect

- You will be kept informed about the progress of the fix.
- If the vulnerability is confirmed, I will issue a patch release and publish a security advisory.
- If the report is declined, I will provide an explanation.
- I will credit reporters in the security advisory (unless you prefer to remain anonymous).

## Security Considerations

### Unsafe Code

Shrew uses `unsafe` Rust in specific, well-scoped areas:

- **`shrew-cuda`**: FFI calls to CUDA driver/runtime (cuBLAS, custom PTX kernels) via `cudarc`.
- **`shrew-cpu`**: SIMD-accelerated operations through `gemm`.
- **`shrew-python`**: PyO3 FFI boundary for Python interop.

All unsafe blocks are documented and minimized.

### Deserialization

- **`.shrew` checkpoints**, **Safetensors**, and **ONNX** model files involve deserialization. Only load models from trusted sources.
- The `.sw` IR parser validates input before execution, but `.sw` files from untrusted sources should be treated with caution.

### Dependencies

I regularly audit the dependency tree. Key dependencies:

| Dependency | Usage |
|------------|-------|
| `cudarc` | CUDA FFI (optional, feature-gated) |
| `pyo3` / `numpy` | Python FFI (optional) |
| `gemm` | SIMD matmul |
| `serde_json` | Checkpoint metadata |

### CUDA Backend

The `shrew-cuda` crate executes GPU kernels and manages device memory. It requires a trusted CUDA toolkit installation. Do not use the CUDA backend with untrusted PTX kernel sources.
