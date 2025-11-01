# AthenaBench

AthenaBench provides cybersecurity benchmarking tasks for evaluating language models on a shared set of CTI tasks. This guide focuses on running the bundled **mini** benchmark, which contains lightweight subsets of each task for quick iteration.

## Run the Mini Benchmark

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure models and credentials** in `athena_eval/config.yaml`. Each entry specifies a provider (`openai`, `gemini`, `huggingface`, or `dummy`) and model name. API keys can be placed in the environment or a `.env` file that is auto-loaded. Example:
   ```
   OPENAI_API_KEY=""
   GEMINI_API_KEY=""
   HF_TOKEN=""
   ```

3. **Generate predictions on the mini subset**
   ```bash
   python -m athena_eval.run --mini --model gpt-4o --task RCM
   ```
   - Omit `--model` or `--task` to iterate over all configured entries.
   - The `--mini` flag swaps each dataset path for its counterpart in `benchmark-mini/` and writes outputs to `runs-mini/<model>/<task>.jsonl`.
   - Evaluation runs by default; add `--no-evaluate` to skip scoring during generation.

4. **Evaluate previously generated predictions (optional)**
   ```bash
   python -m athena_eval.evaluate --mini --model gpt-4o --task RCM
   ```
   This reads predictions from `runs-mini/` and compares them against the mini datasets in `benchmark-mini/`, emitting task-level metrics and refreshing any `*-scored.jsonl` artifacts.

### Mini Tasks Available

Mini splits are provided for every benchmark task and live in `benchmark-mini/`:
- `athena-cti-mcq-3k.jsonl`
- `athena-cti-ate.jsonl`
- `athena-cti-rcm.jsonl`
- `athena-cti-rms.jsonl`
- `athena-cti-taa.jsonl`
- `athena-cti-vsp.jsonl`

Ensure the task names used with `--task` match the keys in `athena_eval/config.yaml` (e.g.,`MCQ3k`, `ATE`, `RCM`, `RMS`, `TAA`, `VSP`).

## Benchmark Results

### Full Benchmark

| Model | CKT (Accuracy) | ATE (Accuracy) | RCM (Accuracy) | RMS (F1-score) | VSP (Acc) | TAA (Accuracy) | Combined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4 | 78.7 | 35.8 | 63.1 | 15.1 | 84.7 | 31.0 | 51.4 |
| GPT-4o | 85.2 | 51.6 | 71.3 | 20.2 | 84.7 | 35.0 | 58.0 |
| GPT-5 | 92.0 | 76.0 | 71.6 | 32.6 | 85.4 | 39.0 | 66.1 |
| Gemini-2.5-flash | 85.1 | 51.6 | 65.1 | 13.4 | 78.5 | 30.0 | 54.0 |
| Gemini-2.5-pro | 89.1 | 76.2 | 71.2 | 28.4 | 85.4 | 31.0 | 63.6 |
| Qwen3-4B | 74.7 | 5.6 | 45.4 | 4.8 | 79.6 | 15.0 | 37.5 |
| Qwen3-8B | 75.7 | 11.8 | 48.9 | 5.5 | 82.6 | 16.0 | 40.1 |
| Qwen3-14B | 78.6 | 19.4 | 54.1 | 7.0 | 80.3 | 17.0 | 42.7 |
| Llama 3.1-8B | 71.8 | 16.4 | 42.8 | 3.6 | 74.0 | 24.0 | 38.8 |
| Llama 3-70b-Instruct | 78.9 | 31.6 | 56.7 | 11.1 | 63.8 | 22.0 | 44.0 |
| Llama 3.3-70b-Instruct | 81.4 | 30.4 | 60.0 | 11.1 | 70.1 | 26.0 | 46.5 |
| Llama-Primus-Merged | 76.3 | 33.8 | 56.6 | 6.6 | 71.9 | 17.0 | 43.7 |

### Mini Benchmark

| Model | CKT (Accuracy) | ATE (Accuracy) | RCM (Accuracy) | RMS (F1-score) | VSP (Acc) | TAA (Accuracy) | Combined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4 | 80.3 | 43.0 | 60.0 | 13.7 | 84.2 | 26.0 | 51.2 |
| GPT-4o | 87.7 | 59.0 | 68.0 | 19.9 | 85.9 | 30.0 | 58.4 |
| GPT-5 | 96.0 | 77.0 | 69.5 | 33.0 | 88.3 | 30.0 | 65.6 |
| Gemini-2.5-flash | 87.7 | 57.0 | 64.5 | 14.0 | 78.3 | 22.0 | 53.9 |
| Gemini-2.5-pro | 91.0 | 77.0 | 68.0 | 29.0 | 86.7 | 24.0 | 62.6 |
| Qwen3-4B | 76.3 | 8.0 | 43.5 | 5.8 | 78.2 | 16.0 | 38.0 |
| Qwen3-8B | 75.3 | 13.0 | 45.5 | 6.8 | 82.6 | 20.0 | 40.5 |
| Qwen3-14B | 82.7 | 21.0 | 49.0 | 8.5 | 78.0 | 16.0 | 42.5 |
| Llama 3.1-8B | 74.0 | 16.0 | 41.0 | 5.4 | 74.1 | 24.0 | 39.1 |
| Llama 3-70b-Instruct | 81.0 | 37.0 | 54.5 | 10.9 | 63.4 | 24.0 | 45.1 |
| Llama 3.3-70b-Instruct | 81.7 | 44.0 | 59.0 | 11.5 | 69.7 | 22.0 | 48.0 |
| Llama-Primus-Merged | 79.7 | 32.0 | 51.0 | 6.4 | 71.8 | 18.0 | 43.1 |


## Access to Full Benchmark
For access to the full benchmark dataset, reach out to info@athenasecuritygrp.com.
