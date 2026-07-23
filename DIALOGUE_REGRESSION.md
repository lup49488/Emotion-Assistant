# Dialogue Regression Suite

`evaluation/dialogue_regression.json` is the source of truth for dialogue
contracts. Add a small, labeled case whenever a production-like issue is fixed
so the same behavior cannot silently regress.

## Default CI checks

The normal test suite validates JSON structure, profile extraction, identity
and language prompt contracts, and the empty-stream retry behavior. These
checks do not call an external model.

```powershell
& D:\software\Python312\python.exe -m pytest test_dialogue_regression.py -q
```

## Opt-in live API smoke checks

Use a dedicated regression account, never a personal account. Live checks can
create memories and usage records for that account and may incur provider cost.
Start the API first, then set only the environment variables below in the
current PowerShell session:

```powershell
$env:DIALOGUE_REGRESSION_URL = 'http://127.0.0.1:8000'
$env:DIALOGUE_REGRESSION_USER_ID = 'serenova-regression'
$env:DIALOGUE_REGRESSION_ACCESS_KEY = 'use-a-dedicated-long-password'
$env:DIALOGUE_REGRESSION_PROVIDER = 'deepseek'
$env:DIALOGUE_REGRESSION_MODEL = 'deepseek-chat'
& D:\software\Python312\python.exe dialogue_regression.py
```

Use `--include-optional` to include the style-RAG and knowledge-RAG cases once
their indexes are ready. The script prints a JSON report and exits non-zero on
a failure, so it can be run manually in CI on a schedule later.
