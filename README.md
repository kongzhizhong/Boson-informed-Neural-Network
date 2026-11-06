# BINN

Code accompanying the paper:

**[A Boson-informed neural network for Ground States of Many-Body Schrödinger equations](https://arxiv.org/abs/2610.18875)**

Zhizhong Kong, Jerry Zhijian Yang, Cheng Yuan, and Xiaofei Zhao (2026).
[arXiv:2610.18875](https://arxiv.org/abs/2610.18875) · [PDF](https://arxiv.org/pdf/2610.18875)

BINN is a neural variational method for ground states of interacting bosons in
continuous space. It incorporates external confinement and short-range pair
structure into the trial wave function, allowing neural networks to represent
the remaining many-body dependence.

The public implementation provides the three-dimensional, **350-particle
harmonic-trap stress test** discussed in Section 4.2.3 of the paper. It runs
variational Monte Carlo (VMC), computes a radial Gross–Pitaevskii (GP) reference,
and generates training diagnostics and a radial-density comparison. The runner
is self-contained and requires no pretrained model or external dataset.

## Method

The trial state is constructed in logarithmic form:

```math
\begin{aligned}
\log \Psi_{\theta}(\mathbf{R})
&= \sum_{i=1}^{N} \chi^{\mathrm{env}}(\mathbf{r}_i)
+ f^{\mathrm{TNN}}(\mathbf{R}) \\
&\quad + \sum_{1 \leq i < j \leq N}
\left[
u_{\mathrm{c}}(r_{ij})
+ \varphi_{\theta_2}(r_{ij}^{2})
\right].
\end{aligned}
```

Its components have distinct roles:

- **Shared positive TNN:** shared orbital functions enforce bosonic permutation
  symmetry, while a log-sum-exp readout combines multiple Hartree-product branches.
- **Potential-aware envelope:** a trainable analytic factor encodes harmonic
  confinement.
- **Cell Jastrow factor:** the positive principal solution of a radial Neumann
  problem supplies the short-range pair structure.
- **Neural pair remainder:** a network with squared-distance input represents
  residual pair correlations.

The local energy uses an exact pooled TNN Laplacian. Configurations are sampled
with the Metropolis-adjusted Langevin algorithm (MALA), and parameters are
optimized by regularized stochastic reconfiguration (SR) in sample space.

## Repository structure

```text
.
├── README.md
└── BINN                 # Standalone Python implementation of the N=350 stress test
```

The Python entry point is named **`BINN`**, without a `.py` extension. All commands
below use this filename.

The public runner covers the single three-dimensional harmonic-trap experiment.
DMC calculations, two-dimensional experiments, architecture ablations, and
external-potential scans from the paper are not included in this runner.

## Get the code

```bash
git clone https://github.com/kongzhizhong/Boson-informed-Neural-Network.git
cd Boson-informed-Neural-Network
```

## Environment

The script declares the following dependencies:

| Dependency | Requirement |
| --- | --- |
| Python | 3.10 or later |
| PyTorch | 2.0 or later, including `torch.func` |
| NumPy | 2.0 or later |
| SciPy | Radial cell solver, interpolation, and root finding |
| Matplotlib | Output figures |

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install a compatible PyTorch build using the
[official installation selector](https://pytorch.org/get-started/locally/).
Choose a CUDA-enabled build for the full GPU experiment. Then install the
remaining dependencies:

```bash
python -m pip install "numpy>=2.0" scipy matplotlib
```

Check the environment:

```bash
python -c "import numpy, scipy, matplotlib, torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

A CUDA-enabled NVIDIA GPU with approximately **16 GB of memory** is recommended
for the default experiment. The reduced smoke check can run on a CPU. Select
mutually compatible package versions; the minimum versions declared by the
script do not constitute a tested compatibility matrix.

## Run the stress test

Run these commands from the repository root.

### 1. Check execution

```bash
python BINN --smoke --device cpu --output-dir results/smoke
```

Smoke mode retains all 350 particles but reduces the walker batch to 4 and the
training run to 2 SR iterations, with shorter warmup and density sampling. It
still solves the cell problem and computes the GP reference. Its outputs check
execution rather than the accuracy of the full experiment.

### 2. Run the full experiment

```bash
python BINN --device cuda --output-dir results/stress_test
```

The default calculation evaluates all 256 walkers and all 61,075 particle pairs
in parallel. Smaller-memory implementations can partition these independent
axes, but chunking requires source changes; no chunk-size command-line option
is provided.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--device` | `auto` | Use `cuda`, `cuda:0`, or `cpu`. Automatic selection uses CUDA when available. |
| `--output-dir` | `binn_3d_stress_output` | Destination for data, the final checkpoint, and figures. |
| `--smoke` | Off | Run the reduced execution check. |

View the command-line help with:

```bash
python BINN --help
```

## Default configuration

Physical and numerical settings are defined in the `Config` dataclass in
[`BINN`](BINN).

| Setting | Default |
| --- | --- |
| Spatial dimension | $d=3$ |
| Particle number | $N=350$ |
| Harmonic frequency | $\omega=1$ |
| Macroscopic coupling | $g_{\mathrm{GP}}=100$ |
| Pair coupling and interaction range | $g=g_{\mathrm{GP}}/N$, $\sigma=10/N$ |
| Cell radius | $\ell=1$ |
| Walker batch | $B=256$ |
| TNN branches | $R=32$ |
| Orbital network | Two hidden layers of width 64, `tanh` activation |
| Neural pair remainder | One hidden layer of width 32, `tanh` activation |
| SR iterations | 2,500 |
| Learning rate | $\eta_t=0.016/(1+t/400)$ |
| SR damping | $10^{-3}$ |
| MALA moves per SR iteration | 5 |
| Random seed | 42 |

The GP reference uses the scaled scattering length $a_0=4.9189$ and effective
coupling $g_{\mathrm{eff}}=4\pi a_0$. If the interaction strength or range scaling
is changed, update `scaled_scattering_length` consistently with the new
interaction before interpreting the GP comparison.

## Outputs

Output paths are relative to the working directory from which the command is
launched. With the full-run command above, the generated files are:

```text
results/stress_test/
├── stress_test_data.npz
├── stress_test_checkpoint.pt
├── stress_test.pdf
└── stress_test.png
```

| File | Contents |
| --- | --- |
| `stress_test_data.npz` | Energy and local-energy CV trajectories, MALA diagnostics, effective SR regularization, radial mass distributions, GP reference, and summary statistics. |
| `stress_test_checkpoint.pt` | Final model state, walkers, configuration, MALA step size, and cell eigenvalue. |
| `stress_test.pdf`, `stress_test.png` | Energy convergence, relative local-energy fluctuation, and the radial mass distribution $4\pi r^2\rho(r)$ compared with GP. |

The terminal also reports the final energy per particle, tail-step standard
deviation, local-energy CV, density $L^1$ distance to GP, and elapsed time.
The tail-step standard deviation describes variation along the training
trajectory; it is not an independent-sample Monte Carlo error estimate.

The checkpoint is saved at the end of the run. Automatic checkpoint loading
and training resumption are not implemented. A new invocation starts a new
calculation and overwrites files with the same names in its output directory.
Use separate output directories for smoke checks and production runs.

## Citation

Please cite the accompanying paper when using BINN in your research:

```bibtex
@misc{kong2026bosoninformed,
  title         = {A Boson-informed neural network for Ground States of Many-Body Schr{\"o}dinger equations},
  author        = {Kong, Zhizhong and Yang, Jerry Zhijian and Yuan, Cheng and Zhao, Xiaofei},
  year          = {2026},
  eprint        = {2610.18875},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  doi           = {10.48550/arXiv.2610.18875},
  url           = {https://arxiv.org/abs/2610.18875}
}
```

## Contact

For questions about the code, contact
[Zhizhong Kong](mailto:zhizhongkong@whu.edu.cn).
