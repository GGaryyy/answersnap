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
pipx install answersnap        # or: uvx answersnap --help
```

Python 3.11+. The latest unreleased code: `pipx install git+https://github.com/GGaryyy/answersnap`.

**Installing gives you the dry run only.** Asking real engines needs your own API keys. `answersnap` ships with no keys and runs no server: every call goes from your machine straight to each provider and is billed to your account there. See [API keys](#api-keys-bring-your-own) below.

## Five-minute dry run (no API keys, no cost)

```bash
answersnap init                                   # writes answersnap.yaml (a fictional coffee brand)
answersnap run --config answersnap.yaml --dry-run
open output/answersnap/example-coffee-co/*/report.html
```

The dry run answers from bundled fictional fixtures and makes no network calls, so you can see the full report before spending anything.

## API keys (bring your own)

A real run calls each engine's API with **your** key. You need a key only for the engines you want; engines without one are skipped, and the report names them.

| Engine | Environment variable | Where to get a key |
|---|---|---|
| Claude | `ANTHROPIC_API_KEY` | [Claude Console](https://platform.claude.com/settings/keys) |
| ChatGPT | `OPENAI_API_KEY` | [OpenAI Platform](https://platform.openai.com/api-keys) |
| Gemini | `GEMINI_API_KEY` (`GOOGLE_API_KEY` also works) | [Google AI Studio](https://aistudio.google.com/apikey) |

API access is billed separately from consumer subscriptions. A ChatGPT Plus, Claude Pro or Gemini app subscription does not include it, so add billing or credits on each platform first.

Set the keys in the shell you run `answersnap` from:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-key"
$env:OPENAI_API_KEY = "your-key"
$env:GEMINI_API_KEY = "your-key"
```

`export` lasts only for that terminal session. To keep the keys, add the lines to `~/.zshrc` or `~/.bashrc`, or set them as user environment variables on Windows. `answersnap` does not read `.env` files.

Keys are read from the environment only, never written to the run directory or the report. Then check which engines are ready:

```bash
answersnap doctor --config answersnap.yaml
```

## Set up your brand

You don't pass a brand on the command line. You describe it, and the questions to ask, in `answersnap.yaml`. Start from the file `answersnap init` writes and replace the fictional coffee brand with your own:

```yaml
schema: 1

brand:
  name: Example Dental
  aliases: ["Example Dental Clinic", "ExampleDental"]   # every spelling an answer might use
  owned_domains: ["example-dental.example"]             # citing these hosts counts as citing you

competitors:                                            # optional; reported side by side
  - name: Sample Smiles
    aliases: []

engines: [anthropic, openai, google]                    # drop the ones you have no key for
repeats: 3                                              # times each question is asked, per engine

recommendation_intents: [recommendation]

prompt_set:
  version: 1                                            # bump it whenever you change any question
  queries:
    - text: "Which clinic is best for dental implants in Taipei?"
      intent: recommendation
    - text: "Is Example Dental worth the price?"
      intent: trust
```

What each part does:

- **Your brand name is never sent to the engines.** Only the `queries` are sent, word for word, so leave your name out of recommendation questions to see whether an engine names you on its own.
- **Aliases matter.** Matching is plain string matching, so an answer that uses a spelling you didn't list counts as not naming you.
- **Questions are yours to write.** `answersnap` does not invent or research them. Write them the way your buyers would ask, in any language.
- **Intents are free-form labels.** Only questions whose intent is listed in `recommendation_intents` count toward the headline "named in recommendation answers".
- **Cost scales with question count.** A run makes questions × `repeats` × engines calls, and prints an estimate before it spends anything.

## A real run

1. Edit `answersnap.yaml` for your brand (see [Set up your brand](#set-up-your-brand)).
2. Set an API key for each engine you want (see [API keys](#api-keys-bring-your-own)). Engines without a key are skipped, and the report names them.
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

## Known limits

- One snapshot per run. Comparing two snapshots is planned; until then, use the intervals: a change smaller than them is not a change.
- The unknown-entity patterns in `metrics/mention.py` are tuned for one category and are not used by the CLI yet.
- Cost figures are list-price estimates, not bills. Gemini reports no search count, so its search charges are not included. A model without a listed price gets no estimate.

## License

Apache-2.0. See [LICENSE](LICENSE).
