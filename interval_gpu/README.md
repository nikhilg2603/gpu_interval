
# File: README.md

# GPU + Auto-Parser + GAOL hybrid example

This example demonstrates:

- A small expression parser (supports x, y variables; numeric constants; + - * /; exp())
- Converting expressions to postfix and building a flattened Node graph
- Uploading Node graph and variable intervals to GPU and evaluating each function in parallel
- Optional GAOL-based refinement on CPU (if built with USE_GAOL=ON)

## Files
- `main.cu` – all code (parser + CUDA kernel + optional GAOL refinement)
- `CMakeLists.txt` – build script

## Build

### Without GAOL (fast start)

```bash
mkdir build && cd build
cmake ..
make -j
./interval_app
```

### With GAOL (rigorous refinement)

1. Install GAOL (system or custom). If installed in a custom prefix set `-DGAOL_ROOT=/path/to/gaol`.

2. Configure CMake:

```bash
mkdir build && cd build
cmake -DUSE_GAOL=ON -DGAOL_ROOT=/path/to/gaol ..
make -j
./interval_app
```

If `cmake` fails to find GAOL, point `GAOL_ROOT` to the installation prefix that contains `include/gaol` and `lib/libgaol.a` (or libgaol.so).


## Notes & Limitations
- GPU interval arithmetic here is *fast* but not IEEE-directed-rounded; it is useful for filtering and approximate bounds.
- GAOL (when enabled) provides mathematically rigorous intervals.
- The parser is minimal — extend it to support more functions (sin, cos, log) and unary minus as needed.


// End of canvas
