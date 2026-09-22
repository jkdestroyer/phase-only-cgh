# Phase-Only Computer-Generated Holography

A Python implementation of phase-only computer-generated holography using the Gerchberg–Saxton (GS) algorithm. The project reconstructs a target image from a phase-only hologram and tracks reconstruction quality as the number of GS iterations increases.

Unlike a typical implementation based on `numpy.fft`, the 2D DFT and inverse DFT are implemented explicitly with Fourier matrices to directly translate the DFT equations into code.

## Method

1. Convert the target grayscale image to normalized intensity and optical amplitude.
2. Initialize a phase-only hologram with unit amplitude and random phase.
3. Propagate between the hologram and image planes using matrix-based 2D DFT/IDFT.
4. Apply the target-amplitude constraint in the image plane and the phase-only constraint in the hologram plane.
5. Repeat the Gerchberg–Saxton update and evaluate reconstruction MSE at selected iterations.

## Results

The reconstruction improves consistently as the number of GS iterations increases.

| Iteration | MSE |
| ---: | ---: |
| 1 | 0.117085 |
| 10 | 0.030823 |
| 50 | 0.006209 |
| 100 | 0.002541 |

### Reconstruction progression

![Reconstruction comparison](results/reconstruction_comparison.png)

### MSE vs. iteration

![MSE graph](results/mse_graph.png)

### Final phase-only hologram

![Final phase hologram](results/phase_hologram_final.png)

## Files

- `dipmain.py` — Gerchberg–Saxton implementation and result generation
- `target.jpg` — target grayscale image
- `results/reconstruction_comparison.png` — target and reconstructed images at iterations 1, 10, 50, and 100
- `results/mse_graph.png` — reconstruction MSE versus iteration count
- `results/mse_results.csv` — numerical MSE results
- `results/phase_hologram_final.png` — final phase-only hologram

## Run

```bash
pip install -r requirements.txt
python dipmain.py
```

The generated outputs are saved in the `results/` directory.
