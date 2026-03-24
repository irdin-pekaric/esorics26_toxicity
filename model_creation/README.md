# Model Evaluation for Toxicity Detection in League of Legends

A concise, reproducible companion for experiments that evaluate toxicity detectors and fine‑tune models on curated League of Legends chat logs.

Summary
-------
This repository contains datasets, evaluation scripts, and finetuning artifacts used to measure and compare toxicity classifiers on in‑game chat. The focus is reproducible evaluation and clear artifacts (predictions + metrics).

Quick start
-----------
Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Requirements
------------
- Ollama must be installed on the machine and the model `llama3.2` must be available locally. Verify installed models with:

```bash
ollama list
```

- You can also try to start the model (this will download it if it's not present):

```bash
ollama run llama3.2
```

- If Ollama is not installed yet, follow the official installation instructions: https://ollama.com/docs

How we use Llama (via Ollama)
-----------------------------
We run `llama3.2` locally through Ollama to evaluate how well a large language model detects toxicity on our chat datasets. Key points:

- Prompting: the evaluation script sends a short, deterministic prompt for each `message` that instructs the model to answer with a single, unambiguous token (for example `TOXIC` or `NOT_TOXIC`, or `TRUE` / `FALSE`). The prompt includes minimal context and explicit output formatting to make parsing reliable.

- Mapping to binary labels: the script maps the model's textual reply to a boolean prediction. Matching keywords (e.g., `TOXIC`, `YES`, `TRUE`, `1`) map to True/toxic; (`NOT_TOXIC`, `NO`, `FALSE`, `0`) map to False/non-toxic.

- Ambiguity & fallback: if the model's reply is ambiguous or doesn't match expectations, a deterministic fallback is applied (we treat as non-toxic by default, and log the raw response for manual inspection).

- Output: per-dataset predictions are written to `2_model_evaluation/evaluations_llama/{dataset}/{adapter}_predictions.csv` (or to `evaluations_llama/` under the running script path). Each predictions CSV contains at least: `id`, `message`, `prediction_text`, `prediction_bool`, and when available, `label`.

- Reproducibility: to reduce variability, prompts are fixed and the script sets deterministic generation parameters where supported by Ollama (e.g., low temperature). Results are saved alongside metrics so you can reproduce the same run on another machine that has Ollama + `llama3.2` installed.

- Dataset-specific prompts: we also provide an alternate evaluation mode that uses dataset‑specific, more detailed prompts (tailored wording, extra context or instructions) when helpful. All outputs for dataset‑specific runs are saved under `2_model_evaluation/evaluations_llama_specified_prompt/{dataset}/{adapter}_predictions.csv`.

All paths used by the scripts are relative to the repository root so you can run them on another machine without embedding user‑specific paths.

Key locations
-------------
- Datasets: `1_dataset/`
- Model evaluation scripts & adapters: `2_model_evaluation/`
- Finetuning artifacts & ONNX exports: `3_finetuning/` and `3_finetuning/*/output/`
- Local evaluation outputs (default):
  - `2_model_evaluation/evaluations/`
  - `2_model_evaluation/evaluations_llama/`
  - For finetuned evaluations (per your request): `4_finetuning_model_evaluation/4_1_finetuning_english_only/evaluations/`

Input CSV format
----------------
Accepts CSVs (comma or semicolon separated) with at minimum:
- `message` — the text to classify
- `label` — binary target (0/1 or truthy values)
- `id` — optional; if missing, scripts synthesize sequential IDs

If your file uses different column names, update the `TEXT_COL` / `LABEL_COL` constants in the evaluation script.

Outputs
-------
Per dataset and adapter the pipeline writes:
- `*_predictions.csv` — per-row predictions and (when available) ground truth
- `metrics_*.json` or aggregated CSVs with accuracy/precision/recall/F1

Summary of evaluation results
-------
The summary of our evaluation results is presented in the `summary_of_evaluation_results.pdf` file. There we provide the evaluation of various models as well as evaluation using Llama 3.1. 

Notes for publication
---------------------
- The repository aims to be reproducible: scripts avoid absolute user paths.
- Ollama can be used to run local LLM evaluations (e.g., `llama3.2`) — ensure Ollama is installed and the model is available on the host.