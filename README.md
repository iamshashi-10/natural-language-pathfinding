# Natural Language Path Finding

Undergraduate project, IIT Kanpur — Shashi Bhidodiya & Krishna Kumayu.

Turn free-form navigation instructions — *"go to the lab, avoid the workshop, stop at
the store room first, stay in the corridor if you can"* — into grid paths that are
**guaranteed walkable**, satisfy the hard constraints, and trade off soft preferences,
without building a formal logic specification.

**The idea:** the language model decides *what* the user wants; A\* decides *how*.
The LLM sees only the instruction and the list of region names and returns a small
JSON *intent form*. It never sees the grid and never writes a coordinate. A planner
over (cell, waypoint progress) then finds the cheapest path that satisfies the form,
and an independent verifier checks every path before it is returned.

## Results

See [`docs/report.md`](docs/report.md) for the write-up and
[`results/tables/big_all.md`](results/tables/big_all.md) for every number.

## Guarantees

[`docs/soundness.md`](docs/soundness.md) proves, for every well-formed intent form:

| | |
|---|---|
| T1 | the path is physically walkable (even if the LLM misread the sentence) |
| T2 | it satisfies every hard constraint of the intent form |
| T3 | `NO_PATH` only if the form is impossible on the map |
| T4 | it is the cheapest such path |
| T5 | soft preferences lengthen it by at most 6× |
| T6 | on the hard constraints the intent form means exactly an LTL formula |

What is *not* guaranteed is that the LLM reads the sentence correctly — that rate is
measured.

## Methods compared

| | |
|---|---|
| M0 | plain A\* to the goal, ignoring the instruction (floor) |
| M1 | end-to-end LLM: sees the ASCII map, writes the path |
| M2 | LLM → LTLf formula → Spot automaton → product planner |
| M3 | **ours**: LLM → intent form → product-graph A\* → verifier |

Ablations: M3-oracle / M2-oracle (gold input), M2-harden (soft preferences made
hard), M3-chain (waypoints chained instead of the product search).

## Reproduce

macOS / Linux, conda (Miniforge), Python 3.11:

```bash
conda create -n nlpath python=3.11 && conda activate nlpath
conda install -c conda-forge spot      # needed for M2 only
pip install -e ".[dev]"
pytest -q                              # all tests, no network
python scripts/run_all.py              # every number, table and figure
```

`run_all.py` reads LLM answers from `cache/llm/` (committed), so it needs **no API
key** to reproduce the reported results. To query the model afresh, put
`GEMINI_API_KEY=...` in `.env` (never committed) and clear the cache; the model is
`gemini-3.5-flash-lite` at temperature 0 (set in `nlpath/llm.py`). A run interrupted
by the free-tier quota resumes from the cache.
`python scripts/run_all.py --llm dry-run` checks the whole harness offline.

The maps and the benchmark are regenerated deterministically by
`scripts/make_maps.py` and `scripts/make_benchmark.py`.

## Layout

```
nlpath/        core: grid, map generator, A*, intent form, compiler, product search,
               verifier, pipeline, LLM client, parser, benchmark, metrics, error analysis
baselines/     M0 plain A*, M1 end-to-end LLM, M2 LTL route, M3-chain
data/maps/     20 maps 40×40 + 10 maps 15×15 (JSON + PNG)
data/instructions/   benchmark: 165 instructions × 3 wordings, + 20 on 15×15 maps
cache/llm/     every LLM answer (for exact reproduction)
results/       raw rows, tables, metrics.json, figures
scripts/       make_maps, make_benchmark, run_all, make_tables, make_figures, check_gemini
docs/          Implementation_Plan.md, soundness.md, report.md, proposal.pdf
tests/         pytest + hypothesis (property tests against a networkx reference)
```
