# Deploying the app (one-time setup, about 15 minutes)

After this setup, **every `git push` to `main` updates the live app automatically.**
GitHub runs the tests first; if they fail, the live app is left untouched.

## 1. Create a Hugging Face account and a write token

1. Sign up at https://huggingface.co (free).
2. Go to **Settings → Access Tokens → Create new token**, type **Write**. Copy it.

## 2. Upload the model weights (once)

From the folder that contains `student_best.pt`:

```bash
pip install huggingface_hub
huggingface-cli login                       # paste the write token
python scripts/upload_weights.py YOUR_HF_USERNAME/hate-speech-distilled-bert
```

The model repo is created **private**, so the weights are not publicly downloadable.

## 3. Connect GitHub to Hugging Face

In the GitHub repo: **Settings → Secrets and variables → Actions**.

| Tab | Name | Value |
|---|---|---|
| Secrets | `HF_TOKEN` | your write token |
| Variables | `HF_SPACE` | `YOUR_HF_USERNAME/hate-speech-detector` |
| Variables | `HSD_MODEL_REPO` | `YOUR_HF_USERNAME/hate-speech-distilled-bert` |

## 4. Deploy

Push any commit to `main` (or open **Actions → Test and deploy → Run workflow**).
The first build takes about 5–8 minutes. Then the app is live at:

```
https://huggingface.co/spaces/YOUR_HF_USERNAME/hate-speech-detector
```

If your Hugging Face username is not `sanjida-khanom`, update the "Live demo" link in `README.md`
and in the portfolio (`HATE_SPEECH_DEMO` in `app/page.js`).

## Day-to-day

- **Change the app:** edit, commit, push. That's it.
- **Add plots from the paper:** fill `confusion_matrix`, `roc_curve`, `model_comparison` or
  `reliability` in `results/metrics.json` and push. The Research page draws them automatically.
- **Check a deployment:** the **Actions** tab on GitHub shows each run; the Space's **Logs** tab
  shows the app starting.

## Notes

- The free CPU Space sleeps after ~48 hours without visitors; the first visit after that takes
  about a minute to wake it.
- The first prediction after a restart is slower because the model is loading. Later requests are fast.
- To change limits without code, add Space variables such as `HSD_MAX_CHARS`, `HSD_MC_PASSES`
  or `HSD_MAX_BATCH_ROWS` (Space **Settings → Variables**).
