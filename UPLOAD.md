# Upload commands (do NOT run until you are logged in)

Replace `OWNER`, `CODE_REPO`, and Hugging Face IDs before running.

## GitHub (code)

```bash
cd release_code
git init
git add .
git commit -m "Public release: matched local64 vs BPE-8k @ 350M"
gh repo create OWNER/CODE_REPO --public --source=. --remote=origin
git push -u origin HEAD
```

Or with an existing empty repo:

```bash
cd release_code
git init
git add .
git commit -m "Public release: matched local64 vs BPE-8k @ 350M"
git remote add origin https://github.com/OWNER/CODE_REPO.git
git push -u origin HEAD
```

## Hugging Face (weights)

```bash
# once per machine
huggingface-cli login

cd release_weights
huggingface-cli upload OWNER/matched350m-local64 matched350m_local64.pt .
huggingface-cli upload OWNER/matched350m-local64 MODEL_CARD_local64.md README.md
huggingface-cli upload OWNER/matched350m-local64 LICENSE SHA256SUMS

huggingface-cli upload OWNER/matched350m-bpe matched350m_bpe.pt .
huggingface-cli upload OWNER/matched350m-bpe MODEL_CARD_bpe.md README.md
huggingface-cli upload OWNER/matched350m-bpe LICENSE SHA256SUMS

# tokenizer (shared)
huggingface-cli upload OWNER/matched350m-bpe-tokenizer tokenizer/bpe.json .
huggingface-cli upload OWNER/matched350m-bpe-tokenizer LICENSE
```

Or create repos in the Hub UI, then:

```bash
huggingface-cli upload OWNER/REPO_ID /path/to/release_weights .
```
