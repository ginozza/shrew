# Contributing to Shrew

This document provides guidelines and instructions for developing, testing, and releasing for the project.

## Development Setup

### Prerequisites

1.  **Rust**: Install the latest stable toolchain via [rustup](https://rustup.rs/).
2.  **Python**: Python 3.8+ is required for the Python bindings and CLI tests.
3.  **CUDA (Optional)**: If you want to build/test with CUDA support, ensure the CUDA Toolkit is installed.

### Setup Steps

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/shrew.git
    cd shrew
    ```

2.  Install development dependencies (if any):
    ```bash
    # For Python tests
    pip install -r requirements.txt # (If present)
    ```

## Workflow

### Building

To build the entire workspace:

```bash
cargo build --workspace
```

### Testing

Run the full test suite:

```bash
cargo test --workspace
```

To run tests with specific features (e.g., CUDA):

```bash
cargo test --workspace --features cuda
```

### Linting & Formatting

Please ensure your code is linted and formatted before submitting a PR:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

## Documentation

We use **mdBook** for the user guide and **standard Rust docs** for the API.

### Building "The Shrew Book"

1.  Install mdBook:
    ```bash
    cargo install mdbook
    ```
2.  Serve locally:
    ```bash
    mdbook serve docs --open
    ```

### Building Rust API Docs

```bash
cargo doc --workspace --no-deps --open
```

## Release Process

We use GitHub Actions to automate releases.

1.  **Version Bump**: Update the version number in `Cargo.toml` files.
2.  **Tag**: Create a new git tag starting with `v` (e.g., `v0.1.0`).
3.  **Push**: Push the tag to GitHub.

```bash
git tag v0.1.0
git push origin v0.1.0
```

The **Release** workflow will automatically:
- Build `shrew-cli` binaries for Windows, Linux, and macOS.
- Package them into archives (`.zip` / `.tar.gz`).
- Create a GitHub Release and upload the assets.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
