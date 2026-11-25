# AgriDoctor AI (GitHub Pages)

Static web app. Plant disease detection runs in the browser with ONNX Runtime Web (v1.22.0, bundled in `ort/`).

## 1. Add your model (required for detection)
From your AgriDoctor project root:

    python make_web_files.py

This creates `model.onnx`, `labels.json` and `kb.json` here. Keep `model.onnx` under 100 MB.
You can also copy those three files by hand (labels.json = your class_names.json).

## 2. Upload to GitHub
Files over 25 MB cannot be added in the browser, so use git:

    git init
    git add .
    git commit -m "AgriDoctor AI"
    git branch -M main
    git remote add origin https://github.com/<you>/<repo>.git
    git push -u origin main

## 3. Turn on Pages
Repo > Settings > Pages > Deploy from a branch > `main` / `(root)` > Save.
The site opens at `https://<you>.github.io/<repo>/`. On a phone, use "Add to Home screen".

## Local test
    python -m http.server 8000      (then open http://localhost:8000)
