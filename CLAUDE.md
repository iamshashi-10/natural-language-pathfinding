# CLAUDE.md — Natural Language Path Finding

Undergraduate project, IIT Kanpur — Shashi Bhidodiya & Krishna Kumayu.

Turn free-form navigation instructions ("go to the lab, avoid the workshop, stop at the
store room first, stay in the corridor if you can") into grid paths that are
**guaranteed walkable**, satisfy hard constraints, and trade off soft preferences —
without building a formal logic specification.

- **Source of truth for design, theorems, evaluation and schedule:**
  `docs/Implementation_Plan.md`. Read it before starting any task.
- `docs/proposal.pdf` — the submitted proposal (motivation and framing only).
- `docs/soundness.md` — full proofs (written on Day 6).

## The one rule

The LLM decides *what* the user wants; A\* decides *how*. In our pipeline the LLM
receives only the instruction and the list of region names, returns an intent form
(JSON), and **never sees the grid or emits a coordinate of the path**. If a change
would let LLM output influence the path other than through the validated intent form,
stop and ask. (Baseline M1 deliberately breaks this rule — keep it in `baselines/`.)

## Invariants the soundness proofs depend on (plan §3)

Do not change any of these silently. If one must change, say so and update plan §3.

- No corner cutting: a diagonal move needs both orthogonal neighbours allowed.
- Edge cost = `step_length(v,u) * weight(u)`; weights are 0.5 (preferred),
  1.0, 3.0 (avoided) — always positive.
- Heuristic = `w_min * octile(v, nearest goal cell)`. Never unscaled octile when
  preferred regions exist (inadmissible → breaks optimality).
- Waypoints are handled by search over product states `(cell, k)`. Sequential
  chaining exists only as the M3-chain ablation.
- Ordering is **strict** (plan §3.1): the sequence is (waypoints…, goal); at
  progress k, cells of elements k+2 onward are blocked. "A before B" means B is never
  entered before A. Planner, the `visits_before` checker and gold LTL must agree.
- Intent forms are validated before planning (names in legend, sequence elements
  pairwise distinct and not forbidden, preferred ∩ avoided = ∅).
- Every path returned by the pipeline passes `verify_path`; a failure raises, never
  returns quietly.
- Start cell in a forbidden region, or in any sequence element except the first
  waypoint → `NO_PATH` with a reason.
- Benchmark `feasible` labels come from the networkx reference solver, never from
  our planner.

## Methods (plan §4.2)

M0 plain A\* (floor) · M1 end-to-end LLM · M2 NL→LTL→automaton planner ·
M3 ours. Ablations: M3-oracle, M2-oracle, M3-chain. M1–M3 use the same LLM, temperature
0, same legend text, same retry budget (≤ 3 calls with error feedback).

## Repository layout (target — create as the days progress)

```
nlpath/              core package
  grid.py            map data structures, load/save JSON
  mapgen.py          room/corridor/doorway generator
  render.py          matplotlib rendering of maps and paths
  astar.py           weighted 8-connected A* (node counting)
  intent.py          intent-form dataclass, schema, validation
  compiler.py        exclusion, weights, waypoint sequence
  product.py         product-graph search (cell, k)
  verify.py          independent verify_path + checkers
  pipeline.py        intent → compile → plan → verify_path (single planning entry point)
  parser.py          LLM → intent form (M3)
  llm.py             the ONLY place that calls an LLM; caching lives here
  prompting.py       shared legend text + retry budget (≤ 3 calls) for M1–M3
  reference.py       networkx product-graph solver (benchmark labels, tests)
  benchmark.py       instruction format, gold LTL (T6), generator
  phrasings.py       templated + hand-written instruction wordings
  evaluate.py        scores any method's output against the gold intent
  metrics.py         Wilson CIs, exact McNemar, bootstrap
  errors.py          failure attribution by stage and type
baselines/
  plain_astar.py     M0
  e2e_llm.py         M1
  ltl_route.py       M2 (Spot: LTLf → DFA → product planner)
  chaining.py        M3-chain ablation
data/maps/  data/instructions/
cache/llm/           cached LLM responses (committed, for reproducibility)
results/             tables and figures
scripts/make_maps.py, make_benchmark.py   regenerate data/
scripts/run_all.py   one command reproduces every number (--llm dry-run: no network)
scripts/make_tables.py, make_figures.py   called by run_all.py
tests/               pytest + hypothesis
docs/
```

## Commands

Environment: macOS, conda env `nlpath` (Miniforge), Python 3.11.

```bash
conda activate nlpath
pip install -e ".[dev]"
pytest -q
python scripts/run_all.py
```

Spot (needed only for M2) comes from conda-forge (`conda install -c conda-forge spot`),
not pip. Keep it an optional import; everything else must work without it.

LLM: Google Gemini API, free tier, `gemini-3.5-flash-lite` — model name in one config place.
Free tier is rate-limited: `llm.py` must rate-limit client-side and back off on 429.

## Conventions

- Python 3.11+, type hints, `dataclasses`, numpy. Tests: pytest + hypothesis;
  `networkx` only in the reference solver (`nlpath/reference.py`: tests and benchmark
  labels), never in a method being evaluated.
- Deterministic: fixed seeds everywhere; maps and instructions stored as JSON.
- All LLM calls go through `nlpath/llm.py`, cached by hash of (model, prompt, params).
  Tests never touch the network — use a fake client. The key is `GEMINI_API_KEY` in
  `.env` (never committed, never printed).
- This is graded work: readable over clever. Short functions, docstrings, and where
  code implements a theorem's assumption, cite it (e.g. `# required by T4`).
- No new dependencies without asking.

## How to work with us

- We work one plan day at a time. Before coding, restate the day's "Done when"
  criterion and propose the files/functions you'll touch; wait for OK on anything
  structural.
- Write tests with the code. Run `pytest` before saying a task is done, and report
  the result honestly (including failures).
- Don't edit `docs/proposal.pdf`. Edit `docs/Implementation_Plan.md` only when asked,
  or when a design decision changes — and tell us.
- Don't commit or push unless asked; suggest a commit message like `day3: product-graph search`.
- At the end of each day, update the Progress section below (one or two lines).

## Settled decisions

- D1 ordering: strict.
- D2 LLM: Gemini free tier, `gemini-3.5-flash-lite` (pinned), temperature 0, same for M1–M3.
- M1 runs on wording 0 only; M2/M3 on all three wordings (~1,300 calls).

## Progress

- Day 0: repo created; plan v2, proposal and this file added.
- Day 1: package skeleton, `grid.py`/`mapgen.py`/`render.py`; 20 maps (40×40, 6 easy /
  7 medium / 7 hard) + 10 small (15×15) in `data/maps/` (JSON, `png/`, overviews) via
  `scripts/make_maps.py`. One name per room; only the main spine is `corridor`. 19 tests pass.
- Day 2: `astar.py` — generic `astar` (reused by Day 3 product search) + `plan_grid`;
  no corner cutting, `w_min`-scaled octile, node counts. Hand-checked cases, a regression
  case for the unscaled-heuristic bug, networkx cross-check. 40 tests pass.
- Day 3: `intent.py` (dataclass + schema + validation, pulled forward from Day 4),
  `compiler.py`, `product.py` (product search via generic `astar`), `baselines/chaining.py`;
  `tests/reference.py` = networkx product-graph solver. 75 tests pass.
- Day 4: `verify.py` (independent of planner: moves, 5 checkers, `hard_checks`, cost) and
  `pipeline.py` (`plan()` verifies every path + reported cost). Known-good/bad path tests;
  injected planner bugs (corner cutting, no ordering) are caught. 113 tests pass.
- Day 5: `tests/test_properties.py` — T1–T4 vs networkx reference, T5 bound, heuristic
  consistency, chaining sound+complete+never better; 1000 random cases each. Fixed
  chaining to use the product's corner rule (`plan_grid(blocked=…)`). 117 tests pass.
- Day 6: `docs/soundness.md` — T1–T6 + Prop. C (chaining sound+complete), plain-language
  proofs; reviewed by an agent, 8 wording fixes applied. T6 holds on corner-separated maps
  (forbidden cells can't be grazed; `mapgen.is_corner_separated`, plan §3.3 updated).
- Day 7: benchmark — 165 instructions × 3 wordings = 495 inputs (135 templated, 30
  hand-written) + 20 on 15×15 maps; gold intent, gold LTL, checks, reference labels/costs.
- Day 8: `llm.py` (Gemini REST via urllib, cache, rate limit, 429 backoff, fake client),
  `prompting.py`, `parser.py` (M3). Not yet run on the real API: no `.env` key.
- Day 9: M2 `baselines/ltl_route.py` (Spot LTLf → DFA → product). T6 cross-check
  M2-oracle == M3-oracle on all 185 instructions + 200 random grids.
- Day 10: M0, M1, `evaluate.py`, `scripts/run_all.py` (dry run end to end, 7 s). 173 tests pass.
- Day 11: full run on Gemini over 3 days of free-tier quota (~1,250 calls, all cached in
  `cache/llm/`); parallel calls under the rate limit; soft instructions re-sampled so the
  preference changes the route (plan §4.1). Results in `results/raw/`.
- Day 12–13: `metrics.py` (Wilson, McNemar, bootstrap), `scripts/make_tables.py`,
  `errors.py` (failure by stage/type) → `results/tables/`, `results/metrics.json`.
- Day 14: `scripts/make_figures.py` → `results/figures/` (validity, M1 vs M3, soft, T5).
- Day 15: `docs/report.md`, `README.md`; `run_all.py` rebuilds tables + figures.
  Headline: M3 100% correct, M2 88%, M1 0% (paths unwalkable), M0 35%. 195 tests pass.
- Wrap-up: `docs/iros/` (IROS-format report, LaTeX + PDF, references checked) and
  `docs/presentation/` (10-min Beamer slides). Next semester: compare with LLM-A*/LLM+P,
  IROS 2027 submission (deadline 1 Mar 2027, tentative).
