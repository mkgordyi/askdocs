# AskDocs

Question answering over the EU AI Act, built the way a production service is built:
tests, a quality gate on control questions, a container image, infrastructure as code
and a pipeline that refuses to deploy if answers got worse.

One rule runs through the whole project: **no source, no answer.** Every answer links
to the article it comes from. If the regulation does not contain the answer, the
service says so instead of guessing.

## Try it

Live demo: https://askdocs-kappa.vercel.app

It is this repository built as a container at a pinned commit and run on Vercel. The
first request after a pause can be slower while the container starts.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/mkgordyi/askdocs)

The codespace installs the dependencies, builds the index and starts the service, about
four minutes the first time. The page opens in the editor's preview tab; if the preview
shows an error, reload it once. The terminal below it shows one JSON log line per question.

## How it works

```
official XHTML  ->  839 citable chunks  ->  index (keywords + embeddings)
                                                   |
question  ->  hybrid search  ->  close enough?  --no-->  "not in the text"
                                      | yes
                                 answer from passages  ->  citation check  ->  JSON log
```

1. **Chunks.** `askdocs/corpus.py` reads the Official Journal XHTML and cuts it by the
   structure of the law: one numbered paragraph is one chunk, a long paragraph is split
   by its points. That is why an answer can cite `Article 5(1)(a)`, not just "Article 5".
2. **Search.** `askdocs/index.py` combines keyword search (BM25, `askdocs/bm25.py`) and
   meaning search (a small local embedding model) with reciprocal rank fusion.
3. **Refusal.** If the closest passage is not similar enough to the question, the
   service refuses. The threshold is not a guess: it is calibrated on control questions.
4. **Answer.** Two modes behind one interface (`askdocs/generate.py`):
   - `extractive` (default): quotes the best passage. No model call, cannot invent text.
   - `llm`: a model writes a short answer from the passages. The answer is accepted only
     if it cites passages that were actually retrieved, otherwise it is dropped.
5. **Log.** One JSON line per question: answered or not, why, similarity, latency,
   tokens, cost. The question text is not logged, only its length and a fingerprint.

## Quality

`evals/golden.jsonl` holds 70 control questions: 54 whose answer is in the regulation
(with the article that must be found) and 16 that the regulation does not answer.
`python -m evals.run` measures the whole flow and compares it with
`evals/thresholds.json`. The pipeline runs it before every deploy.

Search, share of questions where the right article is found:

| search | first result | top 5 | MRR | used |
|---|---|---|---|---|
| keywords only (BM25) | 0.54 | 0.83 | 0.63 | |
| embeddings only | 0.78 | 0.96 | 0.86 | |
| hybrid | 0.83 | 0.96 | 0.88 | yes |
| hybrid + reranker | 0.89 | 0.98 | 0.93 | no: about 1.7 s per question on one CPU |

Whole flow in extractive mode, with the gate values:

| metric | value | gate |
|---|---|---|
| right article in top 5 | 0.96 | at least 0.92 |
| MRR | 0.88 | at least 0.84 |
| answer cites the right article | 0.89 | at least 0.84 |
| out-of-scope questions refused | 0.88 | at least 0.80 |
| in-scope questions refused by mistake | 0.02 | at most 0.06 |

What the numbers do not hide:

- Two out-of-scope questions are answered although they should be refused: GDPR fines
  and the Digital Markets Act. They are close to the topic, and similarity alone cannot
  tell that the text does not answer them. In `llm` mode the model sees the passages and
  can say "not found", which is why the threshold there is only a pre-filter.
- One in-scope question is refused by mistake: it says "chatbot", the regulation says
  "AI systems intended to interact directly with natural persons".
- The refusal threshold was calibrated on the same 70 questions. A separate held-out
  set is the next step.

Reproduce: `python -m evals.run --calibrate`, `python -m evals.experiments.rerank`.

## What has been run, and what has not

| part | status |
|---|---|
| tests and quality gate | run by the pipeline on every push |
| container image | built and run: about 500 MB, about 250 MB of memory in use, starts with networking switched off, runs as a non-root user |
| Azure rollout | template and deploy job are written and `az bicep build` passes; not applied to a subscription yet, so the deploy job is switched off |
| `llm` mode | citation check and refusal paths are unit-tested with a stubbed client; not measured against a live model yet |

## Pipeline

`.github/workflows/pipeline.yml`, three jobs, each runs only if the previous one passed:

1. **quality**: build the index, run the tests, run the quality gate.
2. **image**: build the container image and push it to GitHub Container Registry,
   tagged with the commit.
3. **deploy** (off until the repository variable `DEPLOY_ENABLED` is `true`): apply
   `infra/main.bicep` with the new image, then smoke-test the live service: health, one
   question that must be answered, one that must be refused.

Azure login uses OpenID Connect: a short-lived token per run, no password stored anywhere.

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m askdocs.index          # parse the regulation and build the index
python -m pytest -q
python -m evals.run
uvicorn askdocs.api:app --port 8080
```

Open http://localhost:8080 for the page or http://localhost:8080/docs for the API.

```bash
docker build -t askdocs . && docker run --rm -p 8080:8080 askdocs
```

The index is built into the image, so the container needs no network access at runtime.

## Configuration

| variable | default | meaning |
|---|---|---|
| `ASKDOCS_LLM_PROVIDER` | `none` | `none`, `openai` or `azure_openai` |
| `ASKDOCS_LLM_MODEL` | | model or deployment name |
| `AZURE_OPENAI_ENDPOINT` | | for `azure_openai`; access through the app identity, no key |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | for `azure_openai` |
| `OPENAI_API_KEY` | | for `openai` |
| `ASKDOCS_MIN_CONFIDENCE` | `0.64` without LLM, `0.55` with | refusal threshold |
| `ASKDOCS_TOP_K` | `5` | passages given to the answer step |
| `ASKDOCS_PRICE_IN_PER_1M`, `ASKDOCS_PRICE_OUT_PER_1M` | `0` | token prices for the cost field in logs |

## Deploy to Azure

Not applied to a live subscription yet: the template compiles and the steps below are
the intended one-time setup. It creates a resource group and an identity that GitHub
Actions may use to deploy into that group only.

```bash
RG=askdocs-rg
REPO=mkgordyi/askdocs

az group create --name $RG --location westeurope
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

APP_ID=$(az ad app create --display-name askdocs-github --query appId -o tsv)
az ad sp create --id $APP_ID
az role assignment create --assignee $APP_ID --role Contributor \
  --scope $(az group show --name $RG --query id -o tsv)
az ad app federated-credential create --id $APP_ID --parameters '{
  "name": "github-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:'$REPO':ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

echo "AZURE_CLIENT_ID=$APP_ID"
echo "AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)"
echo "AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)"
```

Then in the repository, Settings, Secrets and variables, Actions, Variables:
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
`AZURE_RESOURCE_GROUP` = `askdocs-rg`, `DEPLOY_ENABLED` = `true`.
To switch on `llm` mode add `LLM_PROVIDER`, `LLM_MODEL` and, for Azure OpenAI, `AZURE_OPENAI_ENDPOINT`.
After the first image is pushed, make the package public (Packages, askdocs, Package
settings, Change visibility) so that Azure can pull it without credentials.

The app scales to zero when idle. The first request after a pause takes a few seconds.

## Monitoring

The service writes one JSON line per question to stdout; Azure Container Apps sends it
to Log Analytics. Refusal rate and latency per hour:

```kusto
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "askdocs-api" and Log_s has '"event": "ask"'
| extend e = parse_json(Log_s)
| summarize questions = count(),
            refused = countif(e.answered == false),
            p95_ms = percentile(todouble(e.latency_ms), 95)
  by bin(TimeGenerated, 1h)
```

## Layout

```
askdocs/      corpus.py  bm25.py  index.py  generate.py  service.py  api.py  static/
evals/        golden.jsonl  thresholds.json  run.py  experiments/
infra/        main.bicep
tests/        corpus, search, answer rules, service, HTTP
data/source/  the regulation as published, see data/NOTICE.md
```

## Next steps

- Measure `llm` mode on the control questions and add answer-level checks to the gate.
- A held-out question set for the refusal threshold.
- The reranker behind a flag, once the latency budget is known.
- Recitals as a second source, shown separately from binding articles.

This is a demo and not legal advice. Source text: Regulation (EU) 2024/1689, © European
Union, see `data/NOTICE.md`.
