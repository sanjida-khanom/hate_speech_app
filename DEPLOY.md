# Deploying the app — Streamlit Community Cloud (free)

Hugging Face now requires a PRO plan for Docker/Gradio Spaces on free CPU, so the
app is deployed on **Streamlit Community Cloud**, which is free and **redeploys
automatically on every push to `main`**. The model weights stay in the private
Hugging Face model repo `sanjidasathi/hate-speech-distilled-bert` and are
downloaded once when the app starts.

## One-time setup (about 10 minutes)

1. Go to https://share.streamlit.io and **Continue with GitHub**. Allow access to
   your repositories.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `sanjida-khanom/hate_speech_app` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL | e.g. `sanjida-hate-speech` (becomes `sanjida-hate-speech.streamlit.app`) |
4. Open **Advanced settings**:
   - **Python version:** 3.11
   - **Secrets:** paste (with your real token):
     ```toml
     HF_TOKEN = "hf_..."
     HSD_MODEL_REPO = "sanjidasathi/hate-speech-distilled-bert"
     ```
5. Click **Deploy**. The first build takes about 5–10 minutes (PyTorch is large).

## Day-to-day

Edit → `git add .` → `git commit -m "..."` → `git push`. GitHub runs the tests,
and Streamlit Cloud rebuilds the app on its own within a minute or two.

## Good to know

- The app sleeps after a period without visitors; the next visitor sees a
  "wake up" button and waits about a minute.
- The first prediction after a restart is slower because the model loads.
- If the app shows **"gone over its resource limits"**, the server ran out of
  memory. Tell the maintainer; the model can be loaded in a lighter mode.
- Change limits without code by adding lines to **Secrets**, e.g.
  `HSD_MC_PASSES = "5"` or `HSD_MAX_CHARS = "500"`.

## Optional: Hugging Face Spaces (needs PRO)

With a Hugging Face PRO plan you can deploy the Docker image instead: set the
repository variable `DEPLOY_TARGET` to `huggingface` (plus the `HF_SPACE`
variable and `HF_TOKEN` secret) and every push will also update the Space.
