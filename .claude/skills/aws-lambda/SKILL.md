---
name: aws-lambda
description: Create the AWS Lambda container example for a Gaussia metric or module under examples/<module>/aws-lambda/, including the dependency and runtime checks that decide whether it can be one.
argument-hint: <module-name>
---

# Gaussia AWS Lambda example

Produce `examples/<module>/aws-lambda/` — eight files, six of them mechanical:

```
handler.py  run.py  requirements.txt  Dockerfile  README.md
scripts/deploy.sh  scripts/update.sh  scripts/cleanup.sh
```

`examples/bestof/aws-lambda/` is the reference for everything except the Dockerfile; take that one from
`examples/agentic/aws-lambda/Dockerfile`, which carries the inline comments.

## 0. The gate — settle this before writing a file

Find the extra this module installs in `[project.optional-dependencies]` of `pyproject.toml`.

**Confirm the extra is declared.** An undeclared extra is not an error: `pip install "<wheel>[<name>]"`
warns, installs nothing extra, and the build succeeds. `examples/runners/aws-lambda/Dockerfile` passes
`MODULE_EXTRA=runners`, which is not in `pyproject.toml`. Hyphens and underscores normalise
(`role_adherence` resolves `role-adherence`), so only a genuinely absent name matters.

**If the extra pulls `torch`, `sentence-transformers`, `transformers`, or an RL stack, stop and choose.**
All five existing examples target an extra that is an empty list, or none at all; not one installs a
deep-learning wheel into a Lambda image, whose ceiling is 10 GB uncompressed, and not one downloads
model weights at cold start.
The options, in the order worth trying:

- **expose the part of the module that needs no extra.** Say in the README which part is absent and why.
  Prove the claim rather than asserting it — import the module in a subprocess with the heavy packages
  blocked on `sys.meta_path` and confirm it succeeds;
- **keep the whole module and measure the built image against the ceiling before writing the rest.**
  If it does not fit, you have learnt that in one step instead of eight;
- **if the module needs no extra at all**, leave `ARG MODULE_EXTRA=` empty and install
  `"${WHEEL}${MODULE_EXTRA:+[$MODULE_EXTRA]}"`, so no extra is requested rather than a heavy one being
  installed for nothing. This diverges from the five existing Dockerfiles by one line; note it in the
  README as deliberate.

**Then check runtime.** Lambda caps an invocation at 15 minutes. A module that loops against a live
external endpoint an unbounded number of times does not belong behind a synchronous HTTP API without
saying so in the README. Prefer a slice whose work is bounded by the payload.

## 1. The six mechanical files

| File | What changes |
|---|---|
| `scripts/deploy.sh` `update.sh` `cleanup.sh` | nothing. Byte-identical across all five examples. |
| `requirements.txt` | nothing. Byte-identical; it pins the four LangChain providers. |
| `handler.py` | line 1 only. 48 lines, otherwise identical across all five. |
| `Dockerfile` | four places, below. |

`handler.py`'s first line follows the subject's kind — `"""AWS Lambda handler for Gaussia <Name>
metric."""` for a metric, `"""AWS Lambda handler for Gaussia <name> module."""` for anything else.

The Dockerfile varies in exactly four places:

1. the build command in the header comment — image name and `-f` path
2. `ARG MODULE_EXTRA=<extra>`
3. `COPY examples/<module>/aws-lambda/requirements.txt .`
4. `COPY examples/<module>/aws-lambda/handler.py examples/<module>/aws-lambda/run.py ${LAMBDA_TASK_ROOT}/`

Leave the numpy/scipy pre-install and the constraints file alone. They are there because the Lambda base
image has no compiler and those are the last versions with manylinux wheels that need none.

## 2. `run.py` — the only file that takes thought

Three parts, in this order.

**`create_llm_connector(connector_config)`** — copy verbatim from `examples/bestof/aws-lambda/run.py`.
It resolves `connector.class_path` by dynamic import, passes `connector.params` to the constructor, and
falls back to the `LLM_API_KEY` environment variable when no `api_key` is given.

**A payload-backed input class.** Metrics subclass `Retriever` and build `Dataset` objects from
`payload["datasets"]` via `Dataset.model_validate`. A module with a different input interface implements
*that* interface against the payload. The invariant is that the function reads its input from the
request body and reaches for nothing external — no S3, no filesystem, no database.

**`run(payload) -> dict[str, Any]`.** The contract every existing example keeps:

- it returns `{"success": False, "error": "<message>"}` for every failure and never raises. An escaped
  exception becomes a 500 with a bare string in `handler.py`, which loses the context;
- it validates before doing work — missing connector, missing input, and whatever else this module
  needs. `bestof` rejects fewer than two datasets and fewer than two distinct `assistant_id`s, which is
  the pattern: reject what the module cannot possibly evaluate, with an error that names the reason;
- the call itself is wrapped and reported as `f"<Name> evaluation failed: {e}"`;
- success returns `{"success": True, ...}` with the emitted fields flattened **explicitly**, field by
  field, not `model_dump()`. The response is an API surface; it should not move when a schema gains a
  field;
- optional knobs come from `payload["config"]` through `config.get(name, default)`. Read every default
  off the signature in `src/` — do not invent them;
- the docstring carries a complete `Example payload:` block. That is what people copy, and it is the
  endpoint's real documentation.

## 3. `README.md`

Base it on `examples/bestof/aws-lambda/README.md` and keep its section order: Description, Invoke URL,
Supported LLM Providers, Test Example (one `###` per provider, each with a working `curl`), Request
Format, Connector Configuration, Module-Specific Fields, Response Format, Response Fields, Error
Responses, Common Errors, View Logs, Deployment Commands, Environment Variables.

Yours to write: the Module-Specific Fields table, the payload inside the curl, the response shape, and
the error table — which lists the errors this module actually returns, including any that come from a
dependency needing a specific provider capability. If step 0 left part of the module out, this file is
where that is stated.

Deployment lines take the module directory name:

```bash
./scripts/deploy.sh <module> us-east-2
aws logs tail "/aws/lambda/gaussia-<module-kebab>" --follow --region us-east-2
```

## 4. Verify

```bash
find examples/<module>/aws-lambda -type f | sort                      # eight files

diff examples/bestof/aws-lambda/requirements.txt examples/<module>/aws-lambda/requirements.txt
for s in deploy update cleanup; do
  diff examples/bestof/aws-lambda/scripts/$s.sh examples/<module>/aws-lambda/scripts/$s.sh
done                                                                  # all four empty

diff examples/bestof/aws-lambda/handler.py examples/<module>/aws-lambda/handler.py
                                                                      # line 1 only

python3 -c 'import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]' \
  examples/<module>/aws-lambda/handler.py examples/<module>/aws-lambda/run.py

uv run ruff check examples/<module>/aws-lambda
```

`examples` is in ruff's `extend-exclude`, so `ruff check .` skips it — but an explicitly passed path is
checked anyway, and all five existing `aws-lambda` directories pass. So this is a real gate, not a
formality.

Then confirm `ARG MODULE_EXTRA` against `[project.optional-dependencies]` one more time, since it is the
line whose mistakes are silent.

Building the image needs Docker and is not part of this task. Say in your report whether it was built:
an unbuilt Dockerfile is a claim, not a result. Same for the deploy scripts — they touch ECR, IAM,
Lambda and API Gateway, so they are the user's to run.
