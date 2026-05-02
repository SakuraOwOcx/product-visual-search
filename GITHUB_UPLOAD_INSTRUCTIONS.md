# GitHub Upload Instructions

This project is prepared for GitHub upload. Large local artifacts are excluded by `.gitignore`, including:

- `data/`
- `models/`
- `outputs/checkpoints/`
- `outputs/indexes/`
- large binary files such as `.pth`, `.safetensors`, `.npy`, `.npz`, `.zip`

## Files Intended for GitHub

The repository should include:

- `product_visual_search_retrieval.ipynb`
- `app.py`
- `README.md`
- `MODEL_AND_DATA_FILES.md`
- `requirements.txt`
- `.gitignore`
- `.gitattributes`
- `scripts/`
- `src/product_search/`
- lightweight reports and metrics under `outputs/reports/`
- selected result figures under `outputs/figures/`

## Install Git

If Git is not installed, install it first:

```powershell
winget install --id Git.Git -e --source winget
```

After installation, restart PowerShell and check:

```powershell
git --version
```

## Upload to GitHub

Create an empty GitHub repository first, then run:

```powershell
git init
git add .
git commit -m "Initial commit: product visual search and retrieval project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

If GitHub asks for authentication, use GitHub login or a personal access token.

## Important

Do not force-add ignored files such as datasets, checkpoints, CLIP weights, or indexes. They are too large for normal GitHub upload and should be regenerated locally using the project scripts.
