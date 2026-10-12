# Running GroundedBI

Use two PowerShell terminals, both starting in the project root (the folder that holds `pyproject.toml`).

## 1. Install Python Dependencies

Run this once, or again after dependencies change:

```powershell
python -m pip install -e ".[dev,api,formats,auth,export]"
```

Set your Gemini key in the environment (`$env:GEMINI_API_KEY`) before indexing or asking questions. The code also reads `.secrets/gemini_key.txt` if that file exists; `.secrets/` is ignored by git.

## 2. Start the Backend

In the first terminal, enable local dev auth and start FastAPI:

```powershell
$env:TRBI_DEV_AUTH_ENABLED="1"
$env:PYTHONPATH="$PWD\src"
uvicorn api.main:app --reload --port 8000
```

The `PYTHONPATH` line makes sure the API imports the local `src\true_rag_bi`
package from this project folder.

Backend URL:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/healthz
```

## 3. Start the Frontend

In a second terminal:

```powershell
cd web
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## 4. Sign In Locally

Open the frontend URL in your browser:

```text
http://localhost:5173
```

Because `TRBI_DEV_AUTH_ENABLED` is on, the app should show:

```text
Continue in local dev mode
```

Click that to enter the GroundedBI analyst console.

## 5. Corpus Dashboard

The Library page manages named corpora. If `data/raw_pdfs/HR Resources` exists, the first corpus points there; otherwise it points at `data/raw_pdfs`. See `data/README.md` for adding your own documents.

Use the Library dashboard to:

- Activate a corpus
- Check drift/stale source status
- Rebuild from the recursive source folder
- Reset generated artifacts without deleting source documents
- Replace the active corpus source path

If the app shows stale old corpus data, use `Reset` on the active corpus and then
`Rebuild`. Rebuild requires a configured Gemini API key because it embeds chunks.

## CLI Alternative

You can also run the RAG pipeline directly from PowerShell:

```powershell
bi-rag index --verbose
bi-rag ask "How much FMLA leave can an employee take?" --debug
bi-rag eval --prompts path\to\prompts.json --verbose
```
