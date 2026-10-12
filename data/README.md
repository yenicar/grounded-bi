# data/

This folder holds your source documents and everything the pipeline builds from them. None of it is committed except this file. The policy PDFs I used are not included; the main README says where they came from.

## Add your own documents

1. Create a folder for the corpus and put your files in it. Subfolders are read recursively. Supported formats: `.pdf`, `.docx`, `.xlsx`, `.csv`.

   ```text
   data/raw_pdfs/HR Resources/
       01_Federal/some-guide.pdf
       02_State/another-policy.pdf
   ```

   If `data/raw_pdfs/HR Resources` exists the first time the app runs, it becomes the default corpus. Otherwise the default corpus reads everything in `data/raw_pdfs/`. Any other folder works too, as long as it is inside the project; add it as a new corpus on the Library page.

2. Set `GEMINI_API_KEY` in your environment. Indexing calls Gemini to embed every chunk and to write a short summary of each document, so it costs API calls in proportion to the size of the corpus.

3. Build the index from the project root:

   ```powershell
   bi-rag index --verbose
   ```

   This runs three stages, each cached by content hash, so a rerun only redoes what changed:
   - **ingest**: extracts each file to Markdown, page by page, and keeps tables as Markdown tables (`bi-rag ingest` runs this stage on its own)
   - **chunk**: splits each document into parent sections of about 800 tokens and child chunks of about 220 tokens
   - **index**: embeds the child chunks into ChromaDB and builds the BM25 keyword index, a term graph and the parent lookup

   In the web app, the Library page's **Rebuild** button runs the same stages as a background job.

4. Ask a question:

   ```powershell
   bi-rag ask "How much FMLA leave can an employee take?" --debug
   ```

## What gets created

```text
data/corpora/registry.json          which corpora exist and which one is active
data/corpora/<corpus-id>/
    normalized_md/                  one Markdown file per source document
    chunks/                         parent and child chunks as JSONL
    index/                          keyword index, graph index, parent lookup, manifest
    chroma/                         vector store (or the folder in TRBI_CHROMA_DIR)
    cache/                          ingestion, chunk, embedding and answer caches
    audit/                          query log
    eval/                           evaluation prompts and reports
```

`Reset` on the Library page deletes these generated files and leaves your source documents alone.

## Running the evaluation

`bi-rag eval` reads questions from `data/corpora/<corpus-id>/eval/default_prompts.json`, or from the file you pass with `--prompts`. No prompts file ships with the repo; write questions that fit your own documents:

```json
{
  "prompts": [
    { "id": "q1", "category": "leave", "question": "How much FMLA leave can an employee take in a year?" }
  ]
}
```

Each run writes `eval_report_<timestamp>.json` and `.md` to the corpus's `eval/` folder.

Scanned PDFs with no text layer come through as empty pages; there is no OCR step yet.
