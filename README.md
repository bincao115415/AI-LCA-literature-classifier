# AI-LCA Literature Classifier

This repository contains a pipeline to classify AI-for-LCA literature using the DeepSeek Chat Completion API and to visualize trends.

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt`

## Usage

Set your API key and run the classifier:

```bash
export DEEPSEEK_API_KEY=YOUR_KEY
python ai_lca_classifier.py --input your_wos.xlsx --output classified.xlsx
```

Common options:

- `--max-workers` to control concurrency
- `--checkpoint-every` to write periodic checkpoints
- `--only-failures-from` to re-run failed rows from a previous output

## Notes

- Classification rules are summarized in `classification rules.rtf`.
- Output files can be large depending on input size.
