# DSAI 490 — Assignment 1

Representation learning on **Medical MNIST** with an **Autoencoder** and a
**Variational Autoencoder**, one of each per anatomical region.

## What gets trained

- 6 per-class AEs and 6 per-class VAEs (one pair per anatomical region: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT).
- 1 global AE and 1 global VAE on all classes combined, used to produce the cross-class latent-space visualization.

14 models total.

## Repository layout

```
.
├── notebook.ipynb         # Colab driver — runs everything top-to-bottom
├── src/
│   ├── config.py          # hyperparameters, class names, paths, seed
│   ├── data.py            # Drive / Kaggle / tf.data pipeline
│   ├── models/
│   │   ├── ae.py          # convolutional AE
│   │   └── vae.py         # convolutional VAE with reparameterization + KL warm-up
│   ├── training.py        # per-class + global training loops
│   ├── viz.py             # reconstruction, latent, sample, denoising plots
│   └── utils.py           # seeding, noise injection, MSE/SSIM metrics
├── figures/               # PNGs produced by the notebook
├── checkpoints/           # trained weights
├── report/
│   ├── report.tex
│   └── report.pdf
├── requirements.txt
└── README.md
```

## Running in Google Colab

1. Upload your `kaggle.json` (from Kaggle → Account → Create New API Token) to Google Drive at `MyDrive/kaggle/kaggle.json`.
2. Open `notebook.ipynb` in Colab.
3. In the "Clone the repo" cell, replace `<user>` in `REPO_URL` with your GitHub username.
4. Run all cells. The notebook:
   - Mounts Drive.
   - Clones this repo into `/content/dsai490-assignment1`.
   - Installs requirements.
   - Downloads the Kaggle dataset to Drive (once; cached on subsequent runs).
   - Extracts it into `/content/medical_mnist/`.
   - Trains all 14 models (~25–40 minutes on a T4 GPU).
   - Produces every figure in `figures/` and a metrics CSV.

## Running locally

```bash
pip install -r requirements.txt
# Place the Kaggle zip at the path configured in src/config.py::Paths.dataset_zip,
# or override it before calling data.ensure_dataset().
jupyter notebook notebook.ipynb
```

## Design notes

- **Latent dimension is 16** for both AE and VAE so comparisons are fair. 2-D latent plots come from PCA / t-SNE projections.
- **KL warm-up** linearly anneals `β` from 0 to 1 over the first 5 epochs to avoid posterior collapse.
- **`tf.data` pipeline** caches decoded images in memory and prefetches — Medical MNIST easily fits in RAM at 64×64 grayscale.
- **Cross-class latent viz uses the global models.** Per-class encoders each have their own unaligned 16-D space; overlaying them would be meaningless.

## Building the report

`report/report.tex` references figures at `../figures/...`, so compile from inside `report/`:

```bash
cd report
pdflatex report.tex && pdflatex report.tex   # second pass for references
```

Or compile from the repo root with a path override:

```bash
pdflatex -output-directory=report report/report.tex
```

## Smoke test before a real training run

To verify the pipeline end-to-end (models build, checkpoints save/reload, every viz helper renders) without needing a GPU or the dataset:

```bash
python -m scripts.smoke
```

Takes under a minute on CPU with TensorFlow installed.

## Deliverables map

| Deliverable | Location |
|---|---|
| Codebase (modular) | `src/` in this repo |
| Experiment notebook | `notebook.ipynb` |
| Figures | `figures/` |
| Technical report | `report/report.pdf` (compiled from `report.tex`) |
| Video demonstration | recorded separately, 2–5 minutes |
