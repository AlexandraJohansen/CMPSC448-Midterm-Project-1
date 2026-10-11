# LLM Authorship Classification: PyTorch CNN vs BiLSTM

## Files
- `dataset_demo.csv`: 54-row **synthetic demonstration dataset** (18 prompts × 3 demo labels). It is not real output from GPT, Claude, Gemini, or any other production LLM.
- `prompts.csv`: 18 reusable prompts to submit to three or more real model families.
- `train.py`: preprocessing, Dataset/DataLoader, CNN and bidirectional LSTM, training, and evaluation.
- `requirements.txt`: Python dependencies.

## 1. Install and run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python train.py --data dataset_demo.csv --epochs 12
```

This run checks that the pipeline works, but metrics on synthetic text are **not valid evidence about real LLM identification**.

## 2. Build the research dataset
1. Take the prompts from `prompts.csv`.
2. Submit the same prompt wording to at least three real LLM families.
3. Save each response as one row in a CSV with columns:
   `prompt_id,topic,LLM_name,LLM_Input,LLM_output,data_status`
4. Use exact model/provider/version identifiers where available, and record collection date and generation settings in a separate metadata file.
5. Set `data_status` to `real_collected` only for genuine responses. Do not relabel synthetic examples as real.
6. Keep the prompt set balanced across models. Use the same system instruction and generation settings where possible; record unavoidable differences.

## 3. Run on collected outputs
Save your real rows to `dataset_real.csv`, then:
```bash
python train.py --data dataset_real.csv --text-col LLM_output --epochs 15
```
Optional research condition using prompt and response together:
```bash
python train.py --data dataset_real.csv --text-col input_plus_output --epochs 15
```
For a fair authorship task, `LLM_output` is the primary condition. Including the prompt may help quantify how much input wording contributes.

## 4. Methodology and limitations
- Split by `prompt_id`, not randomly by row, so responses to the same prompt do not leak across train/test.
- Report accuracy and macro-precision, macro-recall, and macro-F1. Inspect the confusion matrix.
- The sample is small. Use substantially more varied prompts and multiple collection runs if possible. A single held-out split can be unstable; consider repeated grouped splits or grouped cross-validation.
- Do not use test data to tune hyperparameters. If tuning, create a separate validation split grouped by prompt.
- Authorship classification may learn model version, length, formatting, or prompt artifacts rather than robust model identity. Report these caveats.
- If models are updated, treat version/date as part of the experiment. Do not claim results generalize to future versions without testing.
- Discuss bisexuality respectfully, avoid stigmatizing assumptions, and do not treat generated text as evidence about the lived experiences of all bisexual people.

## Research question examples
- RQ1: How accurately can CNN and BiLSTM classifiers identify which LLM generated a response to a held-out prompt?
- RQ2: Does adding the input prompt to the response change attribution performance compared with using the response alone?
- RQ3: Which pairs of model labels are most often confused, according to the confusion matrix?

## Required table for reporting
| Model | Accuracy | Macro-precision | Macro-recall | Macro-F1 |
|---|---:|---:|---:|---:|
| CNN | Fill from `metrics_summary.csv` | | | |
| BiLSTM | Fill from `metrics_summary.csv` | | | |

Do not fill the table with invented metrics; use the results printed by your run.
