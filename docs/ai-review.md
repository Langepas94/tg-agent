# Automatic AI code review

The GitHub Actions pipeline reviews pull requests with an OpenAI-compatible
language model. It is developer tooling and is not linked into the tg-agent
runtime.

## Behavior

The workflow starts for opened, updated, reopened and ready pull requests. It
reads the changed-file list, diff and text snapshots of changed source files
through the GitHub API. It retrieves relevant chunks from the base revision:

- root Markdown files, `docs/`, and scoped agent guides;
- Rust and supporting build or automation code;
- changed source snapshots from the pull request head revision.

The model receives the diff and retrieved context and produces a Markdown review
with potential bugs, architectural problems and recommendations. One pull
request comment is created and updated after later pushes.

The workflow can also be started manually with `workflow_dispatch` and a pull
request number. This is useful for smoke-testing configuration or rerunning a
review without pushing another commit.

## Repository configuration

The default setup uses GitHub Models and needs no repository secret:

- base URL: `https://models.github.ai/inference`;
- model: `openai/gpt-4.1`;
- review language: `Russian`.

The workflow grants the temporary `GITHUB_TOKEN` only `models: read` for model
inference, `contents: read` for repository context and `pull-requests: write`
for its review comment.

To use DeepSeek or another OpenAI-compatible provider, create the Actions secret
`AI_REVIEW_API_KEY` and set `AI_REVIEW_BASE_URL`. `AI_REVIEW_MODEL` and
`AI_REVIEW_LANGUAGE` are optional. For an API whose root includes `/v1`, include
`/v1` in `AI_REVIEW_BASE_URL`.

The workflow needs `contents: read` and `pull-requests: write`. It uses
`pull_request_target`, checks out only the base revision and never executes pull
request code with the model secret. Pull request data is read through the API
and treated as untrusted model input.

## Context limits

The pipeline retrieves at most eight documentation chunks and twelve code
chunks within a shared retrieval budget. Diff, changed source and metadata have
separate limits, and the complete prompt has a hard 48,000-character ceiling.
The changed-file list is always included.

## Local deterministic tests

```bash
python3 -m unittest discover -s .github/scripts -p 'test_*.py'
```
