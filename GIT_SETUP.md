# Git Setup, Push & Sharing Guide

Follow these in order. Total time: ~10 minutes the first time.

## 1. Create the GitHub repo (do this on github.com first)
1. Go to https://github.com/new
2. **Repository name**: `deepfake-detection-suite` (or whatever you like — keep it lowercase, no spaces)
3. **Description** (copy-paste this):
   > Multi-modal deepfake/forgery detection backend — image, document, audio, video, and signature
   > verification, combining pretrained ML classifiers with explainable forensic heuristics. Built for
   > IILM University capstone project (BT-155).
4. **Visibility**: Public (so teammates and Colab can access it without auth headaches) — or Private
   if you'd rather add teammates individually as collaborators.
5. **Do NOT** check "Add a README" / "Add .gitignore" / "Add license" — you already have all of these
   locally, checking them creates conflicts on your first push.
6. Click **Create repository**. GitHub will show you a page with commands — ignore it, use the ones below.

## 2. Push your local project (run these in a terminal, inside the `deepfake_detection_suite` folder)

```bash
cd path/to/deepfake_detection_suite

git init
git add .
git commit -m "Initial commit: 5-module deepfake detection suite + dataset/eval harness"
git branch -M main

# Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub username and repo name:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

If it asks for credentials: GitHub no longer accepts your account password for git operations.
Use a **Personal Access Token** instead:
- github.com → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
- Scope: `repo` → Generate → copy the token
- When `git push` asks for a password, paste the token instead (not your GitHub login password)

Or, simpler: install **GitHub Desktop** (desktop.github.com), sign in there, then "Add local repository"
and click Publish — no token juggling needed.

## 3. Fix the placeholders (one-time, after your first push)

Two places need your real GitHub path (`username/repo-name`):

**a) `README.md`** - the three "Colab notebooks" links near the top: replace `YOUR_USERNAME/YOUR_REPO` in each.

**b) `colab_notebooks/1_run_demo_and_backend.ipynb`** - only if you want Colab to clone the repo instead of uploading a zip:
set `USE_ZIP_UPLOAD = False` and put your repo link in `REPO_URL` (section 2). Uploading the project zip needs no edit.

Then commit and push the fix:
```bash
git add README.md colab_notebooks/
git commit -m "Fix GitHub username placeholders in Colab links"
git push
```

**Never commit** your ngrok token, `08_frontend/.env`, or the fine-tuned model folder `models/` (already in `.gitignore`).
Keep `image_finetuned.zip` in Google Drive.

## 4. Adding your teammates
- Go to your repo on GitHub → Settings → Collaborators → Add people → enter their GitHub usernames or emails.
- They then clone it themselves:
  ```bash
  git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
  ```

## 5. Sharing the link
Once pushed, share **the repo URL** directly:
```
https://github.com/YOUR_USERNAME/YOUR_REPO
```
That single link gives anyone:
- the full code (browsable on GitHub)
- one-click **"Open in Colab"** buttons (the badges in README.md, once you've fixed the placeholders)
- clone/download access for VS Code

For Colab specifically, you can also share a **direct notebook link** without going through the repo page:
```
https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_notebooks/1_run_demo_and_backend.ipynb
```
Paste that straight into a message to your team — clicking it opens the notebook already running in
their own Colab, no download/upload needed at all.

## 6. Keeping it updated after this (normal workflow going forward)
Whenever you or a teammate changes code:
```bash
git add .
git commit -m "Describe what you changed"
git push
```
Whenever you want the latest version someone else pushed:
```bash
git pull
```

## Quick troubleshooting
| Problem | Fix |
|---|---|
| `fatal: not a git repository` | You're not inside the `deepfake_detection_suite` folder — `cd` into it first |
| `remote origin already exists` | You already ran `git remote add origin` once — skip that line, just `git push` |
| Push asks for password and fails | Use a Personal Access Token (step 2) instead of your GitHub password |
| Colab clone fails with 404 | Double-check the repo name/visibility — if it's Private, Colab's `git clone` needs a token too, or make the repo Public |
| Large files rejected on push | Make sure `.gitignore` is actually committed (`git status` shouldn't show `smoke_test_data/` or `*.pt` files as untracked-but-huge) |
