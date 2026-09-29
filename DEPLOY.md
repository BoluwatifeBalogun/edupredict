# Deploying to Render (free tier)

## One-time setup
1. Create a GitHub repo and push this whole folder to it, INCLUDING model/
   (the trained pipeline ships with the code; nothing trains on the server):
       git init && git add -A && git commit -m "YCT performance predictor"
       git branch -M main
       git remote add origin https://github.com/<you>/<repo>.git
       git push -u origin main
2. Sign in at https://render.com with that GitHub account.
3. New -> Web Service -> connect the repo. Render reads render.yaml and
   pre-fills everything (Python, free plan, build + start commands). If it
   asks manually:
       Build command: pip install -r requirements.txt
       Start command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
4. Create Web Service. First build takes ~5 minutes. Your app is then live
   at https://<service-name>.onrender.com - share that link.

## Updating after changes
Commit and push; Render redeploys automatically.

## Free-tier notes
- The service sleeps after ~15 min idle and cold-starts in ~60s. Open the
  URL a few minutes before a demo or defense.
- 512 MB RAM: this build's model is sized to fit (single gunicorn worker).
- To swap in the real Kaggle dataset: put StudentPerformanceFactors.csv in
  data/, run python train.py locally, commit the refreshed model/ folder.
