# answersnap

**Auditable AI-visibility snapshots (GEO).** Ask ChatGPT, Claude and Gemini the questions your buyers ask, keep every answer word for word, and get a report where every number links back to the answers it came from.

Most AI-visibility tools give you one score. `answersnap` gives you the evidence behind it instead:

- **Engines are never averaged.** Claude, ChatGPT and Gemini are reported side by side, each with its own rate.
- **Every rate has its count and a 95% interval.** For example: named in 4 of 9 answers (44%, 95% CI 19–73%). Under 10 answers, the rate is marked *low sample*.
- **Mention, recommendation-question answers and citation are separate columns.** They are not blended into one number.
- **Each answer is asked several times.** One answer is a coin flip.
- **The full answer text is stored as JSON**, one file per answer, along with model ids, dates and a hash of the exact questions.
- **Citation faithfulness.** When an engine returns the text it quoted, `answersnap` fetches the cited page and checks whether the quote is actually on it.
- **"Not observable" is never shown as 0.** When an engine hides its sources, the report says so instead of printing 0%.

## Install

```bash
pipx install git+https://github.com/GGaryyy/answersnap
```

Python 3.11+. (Not on PyPI yet.)

## Five-minute dry run (no API keys, no cost)

```bash
answersnap init                                   # writes answersnap.yaml (a fictional coffee brand)
answersnap run --config answersnap.yaml --dry-run
open output/answersnap/example-coffee-co/*/report.html
```

The dry run answers from bundled fictional fixtures and makes no network calls, so you can see the full report before spending anything.

## A real run

1. Edit `answersnap.yaml`:
   - your brand and every alias it goes by
   - your owned domains
   - your competitors
   - the questions your buyers actually ask
2. Set an API key for each engine you want: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (`GOOGLE_API_KEY` also works). Engines without a key are skipped, and the report names them.
3. Check your setup, then run:

```bash
answersnap doctor --config answersnap.yaml
answersnap run --config answersnap.yaml          # prints the plan and an estimate, then asks before spending
```

The estimate before a run prices one typical recorded call per engine at list price. After a run, each answer is priced from its own reported token usage at the list prices in `pricing.py`, dated in the report. It is an estimate, not a bill: discounts, free allowances and price changes are not reflected, and a model without a listed price gets no estimate.

Useful flags:

| Flag | What it does |
|---|---|
| `--max-calls N` | Caps the spend |
| `--engines anthropic,openai` | Runs a subset of engines |
| `--resume RUN_DIR` | Fills in an interrupted run |
| `--no-fetch` | Skips fetching cited pages |
| `--no-raw` | Does not keep each full API response (`q00_r0.raw.json`) |
| `--yes` | Skips the confirmation prompt (for scripts) |

Rebuild a report from its directory at any time:

```bash
answersnap report output/answersnap/<brand>/<run_id>
```

The report is built only from the run directory. Editing your config afterwards cannot change it.

## What a run directory contains

```
manifest.json          models, dates, prompt-set version + hash, calls planned/made/failed
config.frozen.json     the config as it was when the run started
answers/<engine>/q00_r0.json        answer text, citations, search queries, cited spans, token usage
answers/<engine>/q00_r0.raw.json    the full API response as the SDK returned it (skipped with --no-raw)
answers/<engine>/q00_r0.error.json  a failed call, kept, not hidden
fetched/<hash>.json    cited pages as fetched, for the faithfulness check
report.json / report.md / report.html
```

The raw file is the SDK's JSON form of the response, written with sorted keys. It is not the bytes on the wire. Each answer record names its raw file with a SHA-256 digest, so you can confirm the file has not changed since the run. The parsed record is enough to rebuild the report. The raw response is what lets you re-check the parsing, or extract something new from an old run later. Raw files run to tens of KB per answer.

**Keep run directories private.** They hold full engine output and passages quoted from the pages the engines cited. Do not commit them to a public repository or publish them.

Directories from v0.1 (`answer-1` records) still report. Fields they never captured show as "not recorded", and a resumed v0.1 run lists both record schemas in the report.

## Measurement rules

- **Questions are frozen per version.** Changing any question's text or intent means bumping `prompt_set.version`. `answersnap` refuses to mix answers to different questions under one version.
- **Every call is a fresh conversation.** The question is sent verbatim, with no system prompt and the provider's default temperature, using the consumer-default model tier with web search on.
- **Mention matching is deterministic string matching** on your aliases. No language model decides whether you were mentioned.
- **"Named in recommendation answers" is not "recommended".** It counts answers to recommendation-intent questions that name you, so an answer that names you to warn against you still counts. Read the sentence column. Sentence-level judgment is planned.
- **Faithfulness has no pass/fail threshold.** A quote is either "found verbatim" on the page, or "not found verbatim" with an overlap score. Pages change, render with JavaScript and sit behind paywalls, so a low score is a lead to check, not a verdict.
  - In v0.1 only Claude returns quoted text.
  - ChatGPT and Gemini return links without quotes, so their faithfulness is reported as not observable.
- **Fetching cited pages is guarded.** Every redirect hop is re-checked against private, loopback and link-local addresses, and robots.txt is honoured.

## Engines

| Engine | Model (consumer default) | Citations | Quoted text | Search queries | Cited spans | Token usage |
|---|---|---|---|---|---|---|
| Claude (`anthropic`) | claude-sonnet-5 | yes | yes | yes, with result counts | whole cited passages | yes, plus billed search count |
| ChatGPT (`openai`) | gpt-5.1 | yes | no | yes | character ranges | yes, plus billed search count |
| Gemini (`google`) | gemini-3.7-flash | yes (resolved from redirects) | no | yes | byte ranges, converted and checked against the quoted segment | yes; no search count |

An engine is only allowed to produce data after its parser has handled a real recorded response. See `answersnap verify-providers`. Adapters for Perplexity and xAI exist but are not enabled yet, because they have not passed this check.

## Known limits (v0.1)

- One snapshot per run. Comparing two snapshots is planned; until then, use the intervals: a change smaller than them is not a change.
- The unknown-entity patterns in `metrics/mention.py` are tuned for one category and are not used by the CLI yet.
- Cost figures are list-price estimates, not bills. Gemini reports no search count, so its search charges are not included. A model without a listed price gets no estimate.

## License

Apache-2.0. See [LICENSE](LICENSE).
