# Copilot Actions — GitHub Actions Workflows

Four workflows. All use free GitHub Actions tier (2000 min/month).
Each workflow file goes in `.github/workflows/`.

## ACTION 01 — CI: Lint, Type Check, Import Test

File: `.github/workflows/ci.yml`

```yaml

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
    paths:
      - 'rag-lab/src/**'
      - 'api/**'
      - 'app/**'

jobs:
  lint-and-type-check:
    name: Lint + Types
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dev dependencies
        run: |
          pip install ruff mypy
          pip install -e rag-lab/[dev]

      - name: Ruff lint
        run: ruff check rag-lab/src/ api/

      - name: Mypy type check
        run: mypy rag-lab/src/raglab/ api/ --ignore-missing-imports

      - name: Verify core imports
        run: |
          cd rag-lab
          python -c "
          from raglab.config import Config
          from raglab.types import Document, Question, EvalResult, IntentResult
          from raglab.hooks import get_default_registry
          print('All core imports OK')
          "

  frontend-check:
    name: Frontend Build Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: app/package-lock.json
      - run: cd app && npm ci
      - run: cd app && npm run build
      - run: cd app && npm run lint
```

## ACTION 02 — Data Prep: Download EnterpriseRAG-Bench Slice

File: `.github/workflows/data-prep.yml`

```yaml

on:
  workflow_dispatch:
    inputs:
      source_types:
        description: 'Comma-separated source types to download'
        default: 'confluence,github,jira,slack'
        required: true
      max_docs:
        description: 'Max docs per source type'
        default: '5000'
        required: true

jobs:
  download:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install huggingface-hub datasets

      - name: Download slice
        run: |
          cd rag-lab
          python -c "
          from raglab.parsers.enterprise_bench import download_bench_slice
          source_types = '${{ github.event.inputs.source_types }}'.split(',')
          download_bench_slice(
              source_types=[s.strip() for s in source_types],
              out_dir='corpus/raw/',
              max_docs_per_type=int('${{ github.event.inputs.max_docs }}')
          )
          print('Download complete')
          "

      - name: Upload corpus as artifact
        uses: actions/upload-artifact@v4
        with:
          name: corpus-slice-${{ github.run_id }}
          path: rag-lab/corpus/raw/
          retention-days: 7

      - name: Upload questions.jsonl
        uses: actions/upload-artifact@v4
        with:
          name: golden-questions
          path: rag-lab/golden/questions.jsonl
          retention-days: 30
```

## ACTION 03 — Eval: Run Full Benchmark

File: `.github/workflows/eval.yml`

```yaml

on:
  workflow_dispatch:
    inputs:
      experiment:
        description: 'Experiment folder name'
        default: '02_retrieval_comparison'
        required: true
      index_backend:
        description: 'Index backend: chroma or pageindex'
        default: 'chroma'
        required: true
      max_questions:
        description: 'Number of questions to eval (max 500)'
        default: '50'
        required: true
  push:
    paths:
      - 'rag-lab/experiments/**'
    branches: [main]

jobs:
  run-eval:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install raglab
        run: pip install -e rag-lab/

      - name: Download corpus artifact (if available)
        uses: dawidd6/action-download-artifact@v3
        with:
          name: corpus-slice-*
          path: rag-lab/corpus/raw/
        continue-on-error: true  # OK if no artifact exists yet

      - name: Patch config with inputs
        run: |
          cd rag-lab
          python -c "
          import yaml, sys
          cfg_path = 'experiments/${{ github.event.inputs.experiment }}/config.yaml'
          with open(cfg_path) as f: cfg = yaml.safe_load(f)
          cfg['index']['backend'] = '${{ github.event.inputs.index_backend }}'
          cfg['benchmark']['max_questions'] = int('${{ github.event.inputs.max_questions }}')
          with open(cfg_path, 'w') as f: yaml.dump(cfg, f)
          print(f'Config patched: backend={cfg[\"index\"][\"backend\"]}')
          "

      - name: Run experiment
        run: |
          cd rag-lab
          python -m raglab.run_experiment \
            --config experiments/${{ github.event.inputs.experiment }}/config.yaml

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results-${{ github.run_id }}
          path: |
            rag-lab/out/raglab_out/*.csv
            rag-lab/out/raglab_out/*.md
            rag-lab/out/raglab_out/*.jsonl
          retention-days: 30

      - name: Print summary to job log
        run: |
          cd rag-lab
          python -c "
          import glob, pandas as pd
          files = sorted(glob.glob('out/raglab_out/*_scores.csv'))
          if files:
              df = pd.read_csv(files[-1])
              print(df.groupby(['pipeline','source_type'])['overall_score'].mean().unstack().to_string())
          "
```

### Comment results on PR

```yaml
      - name: Comment results on PR (if PR context)
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const files = fs.readdirSync('rag-lab/out/raglab_out/').filter(f => f.endsWith('_report.md'));
            if (files.length > 0) {
              const report = fs.readFileSync(`rag-lab/out/raglab_out/${files[files.length-1]}`, 'utf8');
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: report.substring(0, 65000)
              });
            }
```

## ACTION 04 — Deploy: Frontend to Vercel + API Health Check

File: `.github/workflows/deploy.yml`

```yaml

on:
  push:
    branches: [main]
    paths:
      - 'app/**'

jobs:
  deploy-frontend:
    name: Deploy to Vercel
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: app/package-lock.json

      - name: Install Vercel CLI
        run: npm install -g vercel@latest

      - name: Pull Vercel environment
        run: |
          cd app
          vercel pull --yes --environment=production \
            --token=${{ secrets.VERCEL_TOKEN }}

      - name: Build project
        run: |
          cd app
          vercel build --prod --token=${{ secrets.VERCEL_TOKEN }}

      - name: Deploy to Vercel
        id: deploy
        run: |
          cd app
          URL=$(vercel deploy --prebuilt --prod \
            --token=${{ secrets.VERCEL_TOKEN }})
          echo "url=$URL" >> $GITHUB_OUTPUT
          echo "Deployed to: $URL"

      - name: Verify deployment
        run: |
          sleep 10
          curl -f "${{ steps.deploy.outputs.url }}" \
            -H "Accept: text/html" --max-time 30
          echo "Deployment verified ✓"

  # Optional: deploy API to Railway (free tier, no credit card for basic use)
  # Uncomment when ready
  # deploy-api:
  #   name: Deploy FastAPI to Railway
  #   runs-on: ubuntu-latest
  #   steps:
  #     - uses: actions/checkout@v4
  #     - uses: bervProject/railway-deploy@main
  #       with:
  #         railway_token: ${{ secrets.RAILWAY_TOKEN }}
  #         service: rag-playground-api

  notify-slack:
    name: Notify on Deploy
    needs: deploy-frontend
    runs-on: ubuntu-latest
    if: always()
    steps:
      - name: Post deploy status
        run: |
          STATUS="${{ needs.deploy-frontend.result }}"
          echo "Deploy status: $STATUS"
          # Add Slack webhook here if desired later
```

## Required GitHub Secrets
Add a `SECRETS.md` (gitignored) documenting required secrets:

- **OPENAI_API_KEY** → OpenAI API key for LLM calls and eval scoring
                     Get from: platform.openai.com/api-keys
                     Cost: GPT-4o-mini ~$0.002 per 50-question eval run

- **VERCEL_TOKEN** → Vercel personal access token for deploy workflow
                     Get from: vercel.com/account/tokens
                     Cost: Free tier supports this project comfortably

- **VERCEL_ORG_ID** → Found in Vercel project settings
- **VERCEL_PROJECT_ID** → Found in Vercel project settings

**Optional** (for Ollama / fully free LLM path):
  No secrets needed — Ollama runs locally on port 11434
  Set `cfg.llm.provider = "ollama"` in config.yaml to use free path

Add to `.gitignore`:
```
  .env
  .env.local
  SECRETS.md
  *.env
```

Add to `rag-lab/.env.example` (committed, safe):
```bash
OPENAI_API_KEY=sk-...your-key-here...
```

## Workflow Run Order (first time setup)    

1. **CI workflow** → runs automatically on every push
2. **Data Prep** → run manually once: Actions tab → "Download Benchmark Data" → Run
                    Input: `source_types=confluence,github,jira,slack`, `max_docs=5000`
3. **Eval workflow** → run manually: `backend=chroma`, `max_questions=50`
                    Then again: `backend=pageindex`, `max_questions=50`
                    Compare the two result CSVs
4. **Deploy** → runs automatically when app/ changes are pushed to main      
