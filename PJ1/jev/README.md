# Jev-guided search experiments

## Package layout and compatibility

```text
jev/
├── core.py   # reusable client, retry, pacing, cache base
├── q2.py     # backward-compatible Q2 imports
└── q6.py     # primary Q6 teaching example and editable rules
```

Existing imports from `jev` and `jev.q2` remain valid. New modules should
import reusable infrastructure from `jev.core`, for example:

```python
from jev.core import JevHeuristicBase
```

This keeps future Q7/Q8 or custom experiments independent of the Q6 module
name while preserving the original Q2 integration API.

The Q2 experiment exposes three heuristic selections. All use the real
TypeSafe Jev decision model through OpenRouter's Decisions API by default.

| Heuristic | Jev usage | Optimality |
|---|---|---|
| `jevNodeHeuristic` | One request for each newly generated unique state | Not guaranteed |
| `jevDirectValueHeuristic` | Choose an exact integer cost for each state | Not guaranteed |
| `jevTieBreakHeuristic` | Batch only unseen states tied at minimum `f` | Preserved |
| `jevPreScoreHeuristic` | Batch every reachable position before A* | Preserved |

## Secure setup

Set the OpenRouter key only in the process environment:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

Do not place keys in source files, committed `.env` files, scripts, reports,
or model artifacts.

Optional settings:

```bash
export JEV_PROVIDER="openrouter"
export JEV_MODEL="typesafe/jev-1.13"
export JEV_BATCH_SIZE=16
export JEV_TIMEOUT_SECONDS=15
```

Direct TypeSafe access remains available with:

```bash
export JEV_PROVIDER="typesafe"
export TYPESAFE_API_KEY="your-typesafe-key"
export JEV_MODEL="jev-latest"
```

The direct provider requires:

```bash
python -m pip install -r pj1-search/jev/requirements.txt
```

## Check gateway connectivity

From the `pj1-search` directory, send one small request through the configured
Jev client:

```bash
python -m jev.check_gateway
```

The command prints the request URL, `answers.check.score`, and `usage`. A failed
request is reported on standard error and exits with a nonzero status. Each run
sends a real request and counts toward API usage.

## Method 1: one request per state

```bash
cd pj1-search
python hero.py -l mediumMaze -p SearchAgent \
  -a fn=astar,heuristic=jevNodeHeuristic
```

For every newly generated unique state, Jev estimates the remaining cost as a
multiple of the Manhattan lower bound. The returned score is converted to:

```text
h_jev = Manhattan * (1 + 0.25 * score)
```

The 10 score levels represent ratios from `1.00x` through `3.25x`. Repeated
states use the cache. This method places Jev's value directly in `g + h`, so
it may overestimate and A* is not guaranteed to return an optimal path.

### Direct integer variant

```bash
python hero.py -l mediumMaze -p SearchAgent \
  -a fn=astar,heuristic=jevDirectValueHeuristic
```

This variant uses a `Choice` question instead of `Score`. Jev receives the
complete wall map, current position, goal, and locally legal directions, but
does not receive Manhattan distance, coordinate deltas, a lower-bound
estimate, or a prescribed pathfinding method. It independently selects one
integer from `0` through `254`; A* uses that integer directly as the heuristic.
The range contains 255 values because that is Jev's maximum number of Choice
options. Manhattan is used only as a local fallback if the API fails.

Real results:

| Map | Requests | API time | Expanded | Path cost |
|---|---:|---:|---:|---:|
| `tinyMaze` | 15 | 3.278 s | 13 | 8 |
| `mediumMaze` | 232 | 41.077 s | 228 | 68 |

On `mediumMaze`, autonomous integer selection preserved the optimal path in
this run but expanded seven more nodes than standard Manhattan and was slower
than the ratio-based per-state method.

## Method 2: on-demand plateau batches

```bash
python hero.py -l mediumMaze -p SearchAgent \
  -a fn=astar,heuristic=jevTieBreakHeuristic
```

The primary priority remains:

```text
f = g + Manhattan
```

When multiple valid OPEN nodes share the same minimum `f`, all unseen states
in that plateau are sent as parallel `Score` questions, up to
`JEV_BATCH_SIZE` per request. Jev selects only within the plateau, so a node
with larger `f` can never pass a node with smaller `f`. With unit step costs
and Manhattan distance, optimality is preserved.

## Method 3: pre-score all reachable positions

```bash
python hero.py -l mediumMaze -p SearchAgent \
  -a fn=astar,heuristic=jevPreScoreHeuristic
```

Before A* starts, an internal BFS enumerates every position reachable from the
start without changing the problem's expansion counter. Positions are sent to
Jev in batches and cached. A* then uses the same admissible primary priority
as Method 2 and reads the Jev second priority locally without further API
calls. This minimizes online decision latency but may score positions that A*
never needs.

## Failure behavior and metrics

If the key, SDK, or API is unavailable, each method prints a warning and uses
a deterministic Manhattan-based fallback. After the first API failure in one
search, a circuit breaker disables further requests for that search.

The final line reports:

```text
[Jev] requests=... scored=... cache_hits=... decisions=... api_time=... failures=...
```

`api_time` is total wall-clock time inside API calls. OpenRouter and TypeSafe
do not publish a fixed latency SLA.

Real OpenRouter Jev 1.13 runs on `tinyMaze` on 2026-09-22 produced:

| Method | Requests | API time | Expanded | Path cost |
|---|---:|---:|---:|---:|
| Per-state heuristic, two runs | 10–12 | 1.487–2.000 s | 8–10 | 8–10 |
| On-demand plateau | 8 | 1.415 s | 14 | 8 |
| Pre-score reachable positions | 1 | 0.249 s | 14 | 8 |

The result demonstrates the expected tradeoff: direct Jev heuristic values can
reduce expansions but its path quality can vary and optimality is not
guaranteed. Using Jev only as a second priority keeps the optimal path.

On the autograder's `mediumMaze`, a real run on 2026-09-22 produced:

| Method | Requests | API time | Expanded | Path cost |
|---|---:|---:|---:|---:|
| Per-state heuristic | 212 | 35.842 s | 219 | 68 |
| On-demand plateau | 143 | 24.878 s | 221 | 68 |
| Pre-score reachable positions | 18 | 3.352 s | 220 | 68 |

The reference autograder solution also has cost 68 and expands 221 nodes with
the standard Manhattan heuristic. Pre-scoring is much faster than the other
Jev methods on this map, while none materially reduces search expansions.

On the course demonstration map `bigMaze`, a real run on 2026-09-22 produced:

| Method | Requests | API time | Expanded | Path cost |
|---|---:|---:|---:|---:|
| Per-state heuristic | 520 | 93.138 s | 497 | 210 |
| On-demand plateau | 283 | 51.942 s | 544 | 210 |
| Pre-score reachable positions | 41 | 8.597 s | 546 | 210 |

The independent standard Manhattan benchmark has cost 210 and expands 549
nodes in under one millisecond. Jev reduced expansions by at most 52 nodes,
but network latency dominated the total runtime.

## Q6: one Jev request per coin-search state

```bash
python hero.py -l tinySearch -p SearchAgent \
  -a fn=astar,prob=CoinSearchProblem,heuristic=jevCoinNodeHeuristic
```

The cache key is the complete Q6 state:

```text
(hero position, remaining coin set)
```

Caching by position alone would be incorrect because returning to the same
cell with a different remaining coin set is a different search problem. Jev
receives the complete wall map, current position, and remaining coin
coordinates, then independently chooses an integer remaining cost from
`0` through `254`. No Manhattan, MST, or prescribed collection order is
included in a successful API request.

The default hard limit is 30,000 API requests per search:

```bash
export JEV_MAX_REQUESTS=30000
export JEV_REQUESTS_PER_SECOND=4
export JEV_MAX_RETRIES=2
```

Every physical HTTP attempt, including retries, counts toward the request
limit. Requests are paced at four per second by default. Timeout, 429, and
transient 5xx responses are retried twice with exponential backoff. After the
request limit or final retry failure, the circuit breaker uses the farthest
remaining coin's Manhattan distance for new states.

Real OpenRouter Jev 1.13 results on 2026-09-26:

| Map | Requests | Cache hits | API time | Expanded | Path cost |
|---|---:|---:|---:|---:|---:|
| `testSearch` | 13 | 0 | 1.832 s | 12 | 7 |
| `tinySearch` | 231 | 0 | 29.851 s | 167 | 27 |
| `trickySearch` | 365 | 1,487 | 66.067 s | 9,712 | 68 |

The known optimal costs are 7, 27, and 60. On `trickySearch`, request 365
timed out, the circuit breaker activated, and the final path cost was 68.
Even without a service failure, direct Jev values are not admissible or
consistent, so this method cannot guarantee an optimal Q6 solution.

A second `trickySearch` run timed out at request 344 after 64.052 seconds of
API time, then expanded 13,010 nodes and returned a path of cost 66. Both runs
processed roughly 340–365 requests in about 50 seconds before the 15-second
read timeout. This repeatable pattern suggests upstream queueing or implicit
throughput throttling under sustained per-state traffic rather than a random
single-request failure, although the endpoint returned no 429 or request ID.

With pacing enabled at four requests per second, a complete third run produced:

| Requests | Retries | Failures | Throttle time | Total API time | Expanded | Path cost |
|---:|---:|---:|---:|---:|---:|---:|
| 4,716 | 0 | 0 | 508.102 s | 1,190.774 s | 7,490 | 68 |

All requests completed, confirming that pacing avoids the earlier sustained
traffic timeout. The run still returned cost 68 instead of the optimal 60, so
the suboptimality comes from the autonomous Jev heuristic values themselves,
not only from the fallback used after a timeout.

### Jev usage on 2026-09-26

Confirmed OpenRouter Jev HTTP attempts recorded by completed commands:

| Activity | Attempts |
|---|---:|
| Q6 `testSearch` | 13 |
| Q6 `tinySearch` | 231 |
| Q6 `trickySearch`, first run | 365 |
| Sustained-load diagnostic probe | 50 |
| Token-usage diagnostic request | 1 |
| Q6 `trickySearch`, second run | 344 |
| Q6 throttled `trickySearch` | 4,716 |
| **Confirmed total** | **5,720** |

Of these attempts, 5,718 returned successful Jev decisions and two ended in
the 15-second read timeout. One earlier throttled run was externally
interrupted before it produced statistics; any requests from that process are
not included in the confirmed total. The OpenRouter key reported
`$0.60757599` of usage for the UTC day after the final run.

## Q6: continuous Score heuristic

This is the recommended single-function teaching example. The public entry
point is:

```python
def jevCoinScoreHeuristic(state, problem):
    ...
```

The prompt and ten ordered score levels are intentionally placed at the top of
`pj1-search/jev/q6.py`, before the API and caching implementation. The function
selected by the command below encapsulates client initialization, complete
state caching, pacing, retries, request limits, statistics, and cleanup.

The same top-of-file teaching block also contains the editable rules for:

- score range and score-to-heuristic conversion;
- complete-state cache identity;
- API-failure fallback;
- provider and model;
- requests per second and timeout;
- retry count and backoff;
- batch size and 30,000-attempt hard limit.

Environment variables can override operational values for experiments without
editing the source. The source defaults remain visible together so students
can copy or modify the example as part of the course design.

```bash
export JEV_REQUESTS_PER_SECOND=5
python hero.py -l trickySearch -p SearchAgent \
  -a fn=astar,prob=CoinSearchProblem,heuristic=jevCoinScoreHeuristic
```

This variant sends the same autonomous Q6 state but uses ten ordered absolute
cost anchors from approximately 0 through 254 moves. Jev returns a continuous
score from 0 through 9, which is mapped to:

```text
h = score * 254 / 9
```

No Manhattan or MST value is included in a successful request. Results:

| Map | HTTP attempts | Retries | Cache hits | API time | Expanded | Path cost |
|---|---:|---:|---:|---:|---:|---:|
| `testSearch` | 9 | 0 | 0 | 1.322 s | 8 | 7 |
| `trickySearch` | 6,226 | 3 | 2,726 | 943.780 s | 8,083 | 60 |

All three transient failures on `trickySearch` succeeded on retry, so the
circuit breaker never activated. The run found the known optimal cost of 60
in 946.3 seconds. It was faster and produced a better path than the 255-option
Choice method, although it expanded more nodes. After this experiment, the
OpenRouter key reported `$0.812384622` of usage for the UTC day.

## Q9: optional Jev state evaluation (ungraded)

This extension is voluntary, does not count toward grades, and does not require
a completed game or a submission. The basic Q9 still uses its original ten-game
autograder and runtime requirements. The reference `multiAgents.py` defaults to
traditional evaluation; the experiment runner enables the Jev hybrid explicitly.

Jev returns one overall board-quality score, mapped from 0..9 to 0..100.
The input and cache key exclude current game score. The traditional evaluator
may still use game score. The score is a heuristic feature, not a calibrated
win probability. The reference experiment blends
`(1-alpha) * traditional + alpha * 10 * jev_score`, with alpha=0.10.
Students can choose their own integration; the student template imposes no
weight, scale, or enable-switch configuration.

### Calling Jev

```python
from jev.q9 import Q9JevEvaluator

evaluator = Q9JevEvaluator()  # Reuse across evaluations for caching.
try:
    jev_score = evaluator.score(state)  # Existing GameState; range 0..100.
    print(evaluator.statistics())
finally:
    evaluator.close()
```

The wrapper reuses the existing client and sends one `overall` Score question.
An illustrative API answer `{"overall": {"score": 6.3}}` becomes `70.0`.
Ordinary calls read `OPENROUTER_API_KEY` from the environment, not automatically
from `.env`. The experiment runner loads `api_key` or `OPENROUTER_API_KEY` from
the project's parent `.env`; an existing nonempty environment key takes precedence.

### One-game experiment

The evaluator and experiment entry point are both in `jev/q9.py`.
Importing the evaluator does not start an experiment. From `pj1-search`, run:

```bash
python -u -m jev.q9
```

Defaults: one traditional baseline followed by one real-Jev game, `smallClassic`,
depth 2, one `RandomDragon`, seed 0, no graphics, fresh Jev cache. Both games reset
to the same seed; different actions may still produce different random trajectories.
Use `--seed` to change the seed, `--weight` to change the reference blend, or
`--games` to opt into additional pairs. The original autograder uses a continuous
random stream, so this paired experiment is not an exact replay of its ten games.
The Jev game opens a status window by default while keeping gameplay graphics
silent. The window shows the current evaluation, returned score and call duration;
it estimates remaining time from the average request rate and an approximate
2,000-request target. Retries count as requests. Use `--no-status-window` to disable it.

Network speed, service load and the many searched states can make a single game
take tens of minutes or longer, with associated API costs. The extension disables
the engine's grading timeout. There is no default experiment budget; optionally run:

```bash
python -u -m jev.q9 --max-seconds 600 --max-requests 300
```

Both budgets apply per Jev game; 0 means unlimited. Request budgets count retries.
Time is checked between evaluations, so an in-flight request/retry may overrun the
time budget. Ctrl+C also stops the experiment and saves the available results.
A forcibly terminated process cannot finalize its report; earlier request logs
and progress checkpoints remain.

The paired-game runner calls `ExpectimaxAgent` and `betterEvaluationFunction` in
`multiAgents.py`, so those methods must be implemented before the full experiment
can run. The student template leaves them as TODOs. The reference Jev integration
uses `USE_JEV`, `JEV_WEIGHT`, `JEV_SCALE` and `_Q9_JEV_EVALUATOR`; the runner
selects the baseline or Jev-enabled evaluation for each paired game. The template
provides defaults for the Jev settings, but students using their own evaluation
implementation must connect the evaluator to that function themselves.

Both Q6 and Q9 share a process-wide rolling-window limiter: no more than five
Jev request attempts can start in any one-second window or 300 in any 60-second
window, including retries. `JEV_REQUESTS_PER_SECOND` can lower this rate; values
above five are capped, and zero does not disable the shared safety ceiling.

### Results

Results are written to `jev/results/q9_overall_TIMESTAMP/`:

- `config.json`: settings, seed, budgets and model.
- `games.json`: paired baseline and Jev results, including partial state on interruption.
- `progress.json`: periodic progress, including during leaf evaluation, and final result.
- `requests.jsonl`: each physical Jev attempt's response, returned usage and duration.
- `billing_before.json` / `billing_after.json`: available key-usage snapshots.
- `summary.json`: completed outcomes, attempt counts, time and available usage totals.

Statuses distinguish won/lost from `api_failed`, `budget_interrupted`,
`user_interrupted`, `error` and `incomplete`. A partial score is not a final game
score; interrupted games are excluded from completed-game outcome averages.
Missing cost/token fields are null rather than zero. Returned usage totals include
only records reporting that field. Key-usage differences can include other activity
or delayed billing and are not necessarily this experiment's exact cost.

The experiment enforces `JEV_Q9_STRICT=1`: API failure after retries stops the game
without silently substituting a local fallback. A budget or service interruption
does not count as a game loss. One game's win/loss cannot establish a reliable win
rate or show that Jev is better; exploratory observations are sufficient.
