# Natural Language Path Finding — Implementation Plan (v2)

**Shashi Bhidodiya (230956) · Krishna Kumayu (230576)**

Internal working document. Contains the full technical design, the formal soundness
guarantee, the evaluation protocol, and the day-by-day schedule with checkpoints.

---

## Changelog (v1 → v2)

The professor's response to the proposal asked for two things:

> "It should be accompanied by a **formal soundness guarantee**, and an evaluation
> should **compare your method with the two alternative approaches**."

| Change | Where | Why |
|---|---|---|
| New section: formal model, theorems T1–T6 with proof sketches, and the exact scope of the guarantee | §3 | Formal soundness guarantee |
| Waypoint *chaining* replaced by search over a **product graph** (cell × waypoint progress). Chaining is kept only as an ablation | §2.4c, §2.5 | Chaining is suboptimal and complicates the proof. The product graph makes the planner sound, complete and optimal for the whole intent form |
| **Bug fix:** the heuristic must be the octile distance **× 0.5** (the cheapest cell weight) | §2.5 | With preferred cells at cost 0.5, unscaled octile distance is *inadmissible*, so the optimality claim would have been false |
| Diagonal corner-cutting is forbidden, and the edge cost is defined exactly | §2.5, §3.1 | The validity theorem needs a precise definition of a legal move |
| New baseline **M2: NL → LTL → automaton product planner** (the formal-specification route) | §4.2 | Comparison with *both* alternative approaches. v1 only had the end-to-end LLM |
| Gold intent forms and gold LTL formulas stored for every instruction | §4.1 | Lets us measure translation accuracy for M2 and M3 separately from planning |
| Oracle-input ablations, infeasible instructions, statistical tests | §4.2–§4.5 | Validates the theorems empirically; makes the comparison fair and quantitative |
| Runtime path verifier + property-based tests against a reference implementation | §3.5 | Makes the guarantee hold for the *code*, not just the algorithm on paper |
| Schedule extended from 10 to 15 working days | §5 | Proofs and the LTL baseline are real work. A 10-day cut list is given |
| **v2.2:** LLM is `gemini-3.5-flash-lite` (D2); M1 runs on one wording per instruction (paraphrase consistency is measured for M2/M3 only) | §4.2, §9 | Free-tier speed; ~1,300 instead of ~1,700 calls |
| **v2.2:** M2-harden leaves preferred regions unconstrained (`G corridor` was unsatisfiable on every instruction) | §4.2 | Found while building M2 (Day 9) |
| **v2.1:** strict ordering semantics (D1 settled), macOS/conda setup, Gemini Flash free tier as the LLM (D2 settled) | §2.4c, §3, §4.1, §6, §9 | Decisions after review |

---

## 1. Problem Statement

### 1.1 The situation

A path planner takes a map, a start cell and a goal cell, and returns the shortest
collision-free path. A\* does this reliably and is guaranteed to find the optimal path
if one exists.

The limitation is the interface. The planner accepts one goal coordinate. It cannot
accept anything else a person might want to say.

### 1.2 What people actually say

> "Go to the lab, but don't cut through the workshop, and stop at the store room on
> the way."

This sentence contains four things:

| Piece | Content | Can plain A\* handle it? |
|---|---|---|
| Destination | the lab | Yes |
| Hard prohibition | not the workshop | No |
| Intermediate stop | the store room | No |
| Ordering | store room *before* lab | No |

Three of the four are lost.

### 1.3 The two existing answers, and why both are unsatisfactory

**Answer A — translate to formal logic.** Convert the sentence into a Linear Temporal
Logic formula, then use a planner that can satisfy formulas.

Problems:
- If the translation is slightly wrong, the robot confidently does the wrong thing.
- The formula is unreadable to the person who gave the instruction, so they cannot
  check whether it was understood correctly.
- LTL is Boolean — a formula is satisfied or it is violated. "Stay in the corridor if
  you can" has no honest Boolean encoding. You would have to convert a matter of
  degree into an arbitrary yes/no threshold, which changes the instruction.

**Answer B — give the map to a language model and ask for the path.** Simple, handles
any phrasing.

Problems:
- Language models are bad at grid-level spatial reasoning. The paths they produce
  routinely pass through walls, skip cells, or stop short of the goal.
- All correctness guarantees are gone. A path that might not be walkable is not a
  worse answer — for a real robot it is a useless one.

### 1.4 The actual problem we are solving

> Given a 2D map with labelled regions and a free-form instruction, produce a path
> that is **guaranteed walkable**, **satisfies the stated constraints**, **respects
> soft preferences proportionately**, and is obtained **without building any formal
> logic specification**.

---

## 2. Solution Design

### 2.1 The core idea

Split the job in two, and never let the halves cross:

> **The language model decides what the user wants. A\* decides how to get there. The
> language model never outputs a single coordinate of the final path.**

Language understanding is what language models are good at. Grid search is what A\* is
good at. Both existing approaches fail because they ask one component to do both jobs.

The structural payoff: since only A\* touches the grid, **a language model error can
make the path wrong with respect to the sentence, but can never make it physically
invalid.** Validity is guaranteed by construction. §3 turns this into a theorem (T1).

### 2.2 Architecture

```
instruction (text)  ─┐
                     ├──▶  [ Stage 1: Parser ]  ──▶  intent form (JSON)
region legend (text)─┘          (LLM)                        │
                                                   [ validation ]
                                                             ▼
grid map ───────────────────────────▶  [ Stage 2: Compiler ] ──▶ allowed cells, weights,
                                          (plain Python)          waypoint sequence, goal set
                                                                     │
                                                                     ▼
                                                   [ Stage 3: A* on product graph ]
                                                                     │
                                                                     ▼
                                                   [ runtime verifier ] ──▶ path or NO_PATH
```

### 2.3 Stage 1 — Parser

The language model receives the instruction and a **plain-text list of region names
only**. It does not receive the grid. It returns a fixed JSON form.

```json
{
  "goal_region": "lab",
  "forbidden_regions": ["workshop"],
  "ordered_waypoints": ["store_room"],
  "preferred_regions": [],
  "avoided_regions": []
}
```

Field meanings:

| Field | Meaning | Hard or soft |
|---|---|---|
| `goal_region` | final destination | hard |
| `forbidden_regions` | must never be entered | hard |
| `ordered_waypoints` | must be visited, in the listed order | hard |
| `preferred_regions` | travel here is encouraged | soft |
| `avoided_regions` | travel here is discouraged but permitted | soft |

Implementation notes:
- Use structured output / JSON mode so the response is always parseable.
- Validate against a schema. Reject and retry (with the validation error) on failure.
- Reject any region name not present in the legend (catches hallucinated rooms).
- **Semantic validation (new):** reject forms where the goal or a waypoint is also
  forbidden, or a region is both preferred and avoided. These are the preconditions
  of the theorems in §3.
- Log every raw response — you will need these for the error analysis.

**Why this is the right intermediate representation:** it is not a formal
specification (no formula, no automaton, no symbolic solver), and a non-expert can
read it and confirm at a glance whether the system understood them.

### 2.4 Stage 2 — Compiler

Pure Python, no learning, no cleverness. Three operations:

**a) Hard exclusion.** Every cell belonging to a forbidden region is marked
untraversable, exactly as if it were a wall.

**b) Soft cost scaling.** Every cell has a base weight of 1.0. Then:
- cells in a preferred region → weight `0.5`
- cells in an avoided region → weight `3.0`

This is how graded preferences are represented. The planner will route through a
preferred corridor when the detour is small and abandon it when the detour is large —
which is exactly what "if you can" means. **This is the thing LTL cannot do, and it
costs us one line of code.** Theorem T5 (§3) bounds how far this can push the path.

**c) Waypoint progress (changed in v2).** Instead of running separate searches
`S → W1 → W2 → G` and concatenating them, the planner searches a product graph whose
states are `(cell, k)`, where `k` is how many elements of the sequence
`(W1, …, Wm, G)` have been entered, in order.

**Ordering is strict:** "visit A before B" means B must not be entered at all until A
has been visited. The goal is the last element of the sequence, so "stop at the store
room, then go to the lab" also forbids passing through the lab before the store room.
In the product graph this is one rule: at progress `k`, entering element `k+1`
advances `k`, and the cells of every later element (`k+2` onward) are blocked. A goal
state is `(cell, m+1)` with `cell` in the goal region.

Why the change:
- Chaining fixes the visiting cell in each waypoint region greedily, so it is not
  optimal even subject to the ordering. The product search is optimal over **all**
  paths that visit the waypoints in order (T4).
- One search, one proof, instead of a concatenation argument.
- The state space is `1600 × (m+1)` for a 40 × 40 grid — trivial.

Chaining is still implemented, as the ablation **M3-chain**, so we can report how much
the product search actually gains.

### 2.5 Stage 3 — A\* search

Standard A\* on the product graph, 8-connected grid.

- **Legal moves:** to any of the 8 neighbours that is free and not forbidden. A
  diagonal move is allowed only if **both** orthogonally adjacent cells are also
  allowed (no corner cutting through walls).
- **Edge cost:** `d(v,u) · w(u)`, where `d` is 1 for orthogonal and √2 for diagonal
  moves and `w(u)` is the weight of the cell being entered.
- **Heuristic:** `h(v,k) = w_min · octile(v, nearest goal cell)`, with `w_min = 0.5`
  when any preferred region exists, else 1.0. **Unscaled octile distance is
  inadmissible once preferred cells cost 0.5** — this was a bug in v1.
  *Implementation:* `w_min` is computed as the smallest weight over allowed cells.
  This equals the value above except when every preferred cell is forbidden, where it
  is 1.0 — still a lower bound on every edge cost, so T4 is unaffected.
- Do **not** use Euclidean distance with diagonal steps at cost 1.
- Instrument it to report nodes expanded.
- If the start cell is inside a forbidden region, or inside a sequence element that
  must come later (anything except the first waypoint), return `NO_PATH` with a reason
  (the robot already violates the instruction). The benchmark places start cells
  outside every region its instruction mentions, so this case is only a guard.

### 2.6 Environment

- Grids of 40 × 40 cells (plus a 15 × 15 set, see risks).
- Generated programmatically: place rectangular rooms, connect with corridors, carve
  doorways. Since you generate them, the region label of every cell is known exactly
  and needs no annotation. Each free cell carries at most one label.
- 20 maps of varying complexity.
- Region vocabulary: `lab`, `workshop`, `store_room`, `print_room`, `corridor`,
  `atrium`, `office`, `stairwell`.
- Save each map as JSON so runs are reproducible.

---

## 3. Formal Model and Soundness Guarantee (new in v2)

This section is what the professor asked for. It will be expanded into
`docs/soundness.md` with full proofs on Day 6. The statements below are the targets;
the proof sketches indicate the argument.

### 3.1 Definitions

- **Grid.** Cells $V = \{0..H-1\} \times \{0..W-1\}$. Free cells $F \subseteq V$.
  Labelling $\lambda : F \to \mathcal{R} \cup \{\bot\}$ (each cell has at most one
  region label). Region $R_r = \{v \in F : \lambda(v) = r\}$.
- **Moves.** For an allowed set $S \subseteq F$, $N_S(v)$ is the set of 8-neighbours
  $u \in S$ of $v$, where a diagonal neighbour counts only if both orthogonally adjacent
  cells are in $S$. Step length $d(v,u) \in \{1, \sqrt{2}\}$.
- **Valid path.** $\pi = (v_0, \dots, v_n)$ is *valid w.r.t. $S$* if every
  $v_i \in S$ and $v_{i+1} \in N_S(v_i)$. *Physically valid* means valid w.r.t. $F$.
- **Intent form.** $I = (g, X, (w_1,\dots,w_m), P, A)$: goal region, forbidden set,
  waypoint sequence, preferred set, avoided set. Write the full sequence as
  $(u_1, \dots, u_{m+1}) = (w_1, \dots, w_m, g)$. **Well-formed** if all names are in
  the legend, $u_1, \dots, u_{m+1}$ are pairwise distinct, none of them is in $X$, and
  $P \cap A = \emptyset$ (enforced by validation, §2.3).
- **Allowed cells.** $S_I = F \setminus \bigcup_{x \in X} R_x$.
- **Admissible start.** $s \in S_I$ and $s \notin R_{u_j}$ for every $j \ge 2$.
- **First entry.** $e_j(\pi) = \min\{ i : v_i \in R_{u_j} \}$ (undefined if never
  entered).
- **$I$-satisfying path.** $\pi$ is valid w.r.t. $S_I$, $v_0 = s$, $v_n \in R_g$, every
  $e_j$ is defined, and $e_1 < e_2 < \dots < e_{m+1}$. This is **strict ordering**:
  element $u_{j+1}$ is not entered at all before $u_j$ has been entered. (Re-entering
  an earlier element later is allowed.)
- **Weights and cost.** $w(v) = 0.5$ if $\lambda(v) \in P$, $3$ if
  $\lambda(v) \in A$, else $1$. Cost $J(\pi) = \sum_i d(v_i, v_{i+1})\, w(v_{i+1})$.
  Length $L(\pi) = \sum_i d(v_i, v_{i+1})$.
- **Hard part.** $I_{hard}$ is $I$ with $P = A = \emptyset$.

### 3.2 Planner, stated precisely

States $(v, k)$ with $k \in \{0..m+1\}$ = number of sequence elements entered so far.
Blocked set at progress $k$: $B_k = \bigcup_{l \ge k+2} R_{u_l}$.
Advance: $\mathrm{adv}(u, k) = k+1$ if $k \le m$ and $u \in R_{u_{k+1}}$, else $k$
(at most one advance per step, since regions are disjoint and elements distinct).
If the start is not admissible, return `NO_PATH`. Initial state
$(s, \mathrm{adv}(s, 0))$. Edges $(v,k) \to (u, \mathrm{adv}(u,k))$ for
$u \in N_{S_I}(v) \setminus B_k$, with cost $d(v,u)\,w(u)$. Goal states $(v, m+1)$
with $v \in R_g$. Search with A\*, heuristic $h(v,k) = w_{min} \cdot \min_{t \in R_g}
\mathrm{octile}(v,t)$, closed set. Return the projection of the found state path to
cells, or `NO_PATH`.

### 3.3 Theorems

**T1 — Physical validity (unconditional).** For *every* well-formed intent form —
whether or not it is a correct reading of the instruction — the pipeline returns
either a physically valid path starting at $s$, or `NO_PATH`.

*Sketch.* The returned path is reconstructed from A\* parent pointers; every edge was
generated by $N_{S_I}$, and $S_I \subseteq F$. The LLM's output can only shrink $S_I$,
change positive weights, and choose the goal/waypoint sets; none of these can create an
edge outside $F$. ∎
*Corollary:* no parser error, of any kind, can produce a path through a wall.

**T2 — Soundness w.r.t. the intent form.** Any returned path is $I$-satisfying.

*Sketch.* Edges stay in $S_I$ (so no forbidden cell). Invariant: at progress $k$, no
cell of $u_j$ for $j > k$ has been visited — $u_{k+1}$ entry advances $k$, and all
later elements lie in $B_k$, which no edge enters. Hence the first entry of $u_j$ is
exactly the step where $k$ becomes $j$, so $e_1 < \dots < e_{m+1}$. The terminal
state has $k = m+1$ and $v \in R_g$. ∎

**T3 — Completeness.** If the start is admissible and any $I$-satisfying path exists,
the planner does not return `NO_PATH`. Equivalently, `NO_PATH` is a proof that the
(parsed) instruction is infeasible on this map.

*Sketch.* Lift any $I$-satisfying path to the product graph by setting $k_i$ = number
of elements whose first entry is at or before step $i$. Strict ordering means that at
each step the next cell is not in $B_{k_i}$, so every lifted edge exists; the lift ends
in a goal state. (Under strict semantics the lift is unique — no matching argument is
needed.) The product graph is finite and A\* with a closed set is complete on finite
graphs. ∎

**T4 — Optimality.** The returned path minimises $J$ over all $I$-satisfying paths.

*Sketch.* Lifting (T3) and projection (T2) are cost-preserving, so the optimal product
path projects to an optimal $I$-satisfying path. The heuristic is consistent: octile
distance is a lower bound on step length between 8-connected cells and is 1-Lipschitz
under single moves, a minimum over targets preserves this, and every edge costs at
least $w_{min}\, d(v,u)$. A\* with a consistent heuristic and a closed set returns an
optimal path. ∎
*Note:* this is exactly the statement that breaks if the heuristic is not scaled by
$w_{min}$.

**T5 — Bounded cost of compliance for soft preferences.** Let $L^*$ be the length of
the shortest $I_{hard}$-satisfying path and $\hat\pi$ the returned path. Then

$$L(\hat\pi) \;\le\; \frac{w_{max}}{w_{min}}\, L^* \;\le\; 6\, L^*,$$

and if $P = A = \emptyset$, $\hat\pi$ is a shortest $I_{hard}$-satisfying path.

*Sketch.* Soft preferences do not change the feasible set, so the shortest
$I_{hard}$-path $\pi_L$ is also $I$-satisfying. By T4,
$w_{min} L(\hat\pi) \le J(\hat\pi) \le J(\pi_L) \le w_{max} L^*$. ∎
*Interpretation:* this is the formal content of "if you can". Soft preferences can
never cause an unbounded detour, and the bound is set directly by the two multipliers.
No Boolean formula expresses this trade-off.

**T6 — Correspondence with LTL on the hard fragment.** Define the finite-trace (LTLf)
formula over region propositions, using strong until $\mathbf{U}$:

$$\psi_{m+1} = \mathbf{F}(g \wedge \mathit{last}), \qquad
\psi_j = \neg(u_{j+1} \vee \dots \vee u_{m+1})\;\mathbf{U}\;(u_j \wedge \psi_{j+1})
\quad (j = m, \dots, 1),$$

$$\varphi_I \;=\; \mathbf{G}\,\neg(x_1 \vee \dots \vee x_k) \;\wedge\; \psi_1
\qquad (\text{if } m = 0:\ \varphi_I = \mathbf{G}\,\neg(\dots) \wedge \psi_{1} \text{ with } \psi_1 = \mathbf{F}(g \wedge \mathit{last})).$$

Example — "visit A before B, then the lab, avoid the workshop":
`G !workshop & ((!(B | lab)) U (A & ((!lab) U (B & F(lab & last)))))`.

Then for an admissible start, $\pi$ is $I_{hard}$-satisfying iff its label trace
satisfies $\varphi_I$ — **on corner-separated maps** (no legal diagonal step passes the
corner of a region cell without entering that region). The condition is needed
because forbidden cells are treated as walls, so a path may not graze them
diagonally, and a formula over visited cells cannot express "not grazed". All
generated maps are corner-separated (`mapgen.is_corner_separated`); see
`docs/soundness.md` §6.
Consequently, with identical edge costs, our planner and an automaton-product planner
for $\varphi_I$ return paths of equal optimal cost.

*Sketch.* Direct unfolding of LTLf semantics against the definition in §3.1. ∎
*Uses:* (a) generates the gold LTL formula for every instruction, to score the LTL
baseline's translations; (b) a cross-check test between the two planners;
(c) shows the intent form is *at least as expressive* as the LTL fragment needed for
the hard classes, and strictly more expressive once soft preferences are added.
$\varphi_I$ appears only in the proof and in evaluation — **the system never constructs
it**, so proposal clause (iv) still holds.

### 3.4 Scope of the guarantee — state this precisely

```
instruction ──(parser: NOT guaranteed, measured)──▶ intent form ──(T1–T5: guaranteed)──▶ path
```

- **Unconditional:** physical validity (T1). Holds even when the parser is wrong.
- **Conditional on a correct parse:** hard-constraint satisfaction, completeness,
  optimality, bounded detour (T2–T5).
- **Not guaranteed:** that the parser reads the sentence correctly. Therefore
  P(path satisfies the instruction) ≥ P(parse is correct), and the evaluation measures
  the parse accuracy directly.
- The LTL route has *exactly the same structure* (its guarantees are conditional on a
  correct formula). The end-to-end LLM route has no guarantee at either level. So the
  comparison in §4 reduces to: translation accuracy, expressiveness (soft
  preferences), readability, and cost.

Saying this clearly is stronger than overclaiming. A professor will check whether the
"soundness guarantee" secretly assumes the LLM is correct; here the assumption is
explicit and its failure rate is measured.

### 3.5 Making the guarantee hold for the code

The theorems are about the algorithm. Two mechanisms carry them to the implementation:

1. **Runtime verifier.** `verify_path(path, map, intent)` independently checks physical
   validity and $I$-satisfaction (written separately from the planner). Every pipeline
   output passes through it; a failure raises an error and is counted as a bug, never
   silently returned.
2. **Property-based tests** (`hypothesis`) on random small grids (≤ 10 × 10) and random
   well-formed intent forms:
   - planner optimal cost == cost from a reference shortest path on an explicitly
     built product graph (`networkx`), including the `NO_PATH` cases (T3, T4);
   - every returned path passes `verify_path` (T1, T2);
   - T5 inequality holds on every case;
   - for hard-only forms, our cost == automaton-product planner cost on $\varphi_I$ (T6).

---

## 4. Evaluation (expanded in v2)

### 4.1 Benchmark

Roughly 165 instructions across six categories, each with **two paraphrases** (so
~495 inputs). Generate the bulk from templates over region names, then hand-write about
30 in deliberately natural, messy phrasing.

| Category | Example | Count |
|---|---|---|
| Plain destination | "Go to the lab." | 20 |
| Prohibition | "Reach the lab without entering the workshop." | 35 |
| Intermediate stop | "Stop at the store room, then go to the lab." | 30 |
| Ordering | "Visit the store room before the print room, then the lab." | 30 |
| Soft preference | "Go to the lab, and stay in the main corridor if you can." | 35 |
| **Infeasible (new)** | goal reachable only through the forbidden region | 15 |

Each instruction is stored with machine-checkable requirements **and gold
representations** (gold intent from the template; gold LTL generated from it via T6;
hand-written ones get hand-written gold intent):

```json
{
  "text": "Reach the lab without entering the workshop.",
  "map_id": "map_07",
  "start": [3, 4],
  "feasible": true,
  "gold_intent": {"goal_region": "lab", "forbidden_regions": ["workshop"],
                  "ordered_waypoints": [], "preferred_regions": [], "avoided_regions": []},
  "gold_ltl": "G !workshop & F(lab & last)",
  "checks": [
    {"type": "ends_in", "region": "lab"},
    {"type": "never_enters", "region": "workshop"}
  ]
}
```

Checker types: `ends_in`, `never_enters`, `visits`, `visits_before`,
`fraction_in_region` (reported as a number, not pass/fail). `visits_before(A, B)` is
strict, as in §3.1: A is entered, and B is not entered at any step before A's first
entry.

Generator rules that follow from strict ordering:
- Start cells lie outside every region the instruction mentions.
- The `feasible` flag is computed with the reference solver (`networkx` on the
  explicit product graph), not with our planner, so the label is independent of the
  thing being evaluated. Strict ordering makes more templated instructions
  infeasible; feasible ones are kept for the five main categories and infeasible ones
  feed the infeasible category.

### 4.2 Methods compared

| ID | Method | Represents |
|---|---|---|
| M0 | Plain A\* start → goal, ignores the instruction | Floor |
| M1 | **End-to-end LLM:** ASCII grid + instruction, model outputs coordinate list | Alternative B (proposal §3.2) |
| M2 | **NL → LTL → planner:** LLM translates to an LTLf formula over region names; syntax-checked with Spot; formula → DFA (Spot); product with the grid; shortest path with the same edge costs | Alternative A (proposal §3.1) |
| M3 | **Ours:** intent form + product-graph A\* | Proposed |

Fairness rules for M1–M3:
- Same LLM, temperature 0, same region legend text, same few-shot budget.
- Same retry budget: up to 3 calls, each retry given the validator / syntax-checker
  error message. Report calls used.
- Soft preferences in M2: the LLM is free to encode them however it likes (we report
  what it does). Additionally, on the soft category we run two fixed oracle encodings
  — **M2-drop** (ignore the preference) and **M2-harden** (an avoided region becomes
  `G !region`; a preferred region is left free, i.e. the path uses it or not,
  whichever is shortest) — to show the two options LTL actually has.
  *(v2.2: the earlier `G corridor` encoding of "prefer the corridor" was
  unsatisfiable on every instruction, since start cells lie outside the corridor.)*

Ablations:
- **M3-oracle / M2-oracle:** gold intent / gold LTL fed directly to the planners.
  Isolates translation error; M3-oracle must score 100% on hard checks (empirical check
  of T2–T4).
- **M3-chain:** sequential waypoint chaining instead of the product graph — measures
  the optimality gain of §2.4c.

### 4.3 Metrics

| Metric | Definition | Expected |
|---|---|---|
| Path validity | walkable (T1 sense) and ends in the goal | 100% for M0, M2, M3 by construction; **low for M1 — headline** |
| Hard constraint satisfaction | fraction passing all hard checks | M0 low; M1 low/mid; M2 and M3 high, limited by translation |
| Translation accuracy | M3: parsed form == gold intent. M2: parsed formula *equivalent* to gold φ (Spot `are_equivalent`) | Compares the two interpretation stages directly |
| M2 syntax-failure rate | formula unparseable after retries | Non-zero; ours has no syntax to get wrong beyond JSON |
| Infeasibility detection | on the infeasible category: returns `NO_PATH` (correct) vs returns a "path" | M2, M3 correct by T3; M1 will invent paths |
| Cost ratio | our cost ÷ cost of best instruction-satisfying path (from M3-oracle) | ≈ 1.0 |
| Soft compliance | fraction of path in preferred region and length ratio vs M3-oracle, per method | M2-drop ignores; M2-harden over-constrains or fails; M3 trades off |
| Paraphrase consistency | do the 3 wordings yield the same parsed form / equivalent formula? | Brittleness, both M2 and M3 |
| LLM calls, latency | per instruction | — |
| Nodes expanded | A\* work | Cost of constraints; M2 vs M3 |

### 4.4 Statistics

- Wilson 95% confidence intervals on every rate.
- Paired McNemar tests, M3 vs M2 and M3 vs M1, on hard satisfaction (same instructions).
- Bootstrap CIs on cost ratios.
- Report per category, not only aggregate — the interesting differences are in the
  soft and infeasible categories.

### 4.5 Expected results, honestly stated

- M1 produces valid paths well under 50% of the time on 40 × 40; M3 100% (T1).
- **On the hard categories, M2 and M3 may well tie.** That is fine and should be said
  plainly: T6 shows they plan over the same constraint. The differences we expect to
  show are (i) soft preferences, where M2 must drop or harden, (ii) translation /
  syntax failures, (iii) readability of the intermediate form.
- Optional, if time allows: a small readability check with 8–10 classmates — show the
  instruction plus either the intent form or the LTL formula, ask "does this match?",
  record accuracy and time.

### 4.6 Failure categories for error analysis

- Region name in instruction not matched to any map region
- Constraint entirely missed by the parser
- Soft preference misread as a hard prohibition (or vice versa)
- Ordering constraint dropped or inverted
- (M2) Syntactically invalid formula / semantically wrong temporal operator
- Correctly parsed but infeasible — `NO_PATH` correctly returned
- Correctly parsed, path found, but a soft preference was poorly honoured
- (M1) Path invalid: wall crossing / discontinuity / stops short

---

## 5. Schedule and Checkpoints (15 working days)

### Week 1 — Planner and formal core

| Day | Task | Done when |
|---|---|---|
| 1 | Package skeleton, grid representation, map generator, matplotlib rendering | 20 maps (+ 15×15 set) generated, saved as JSON, rendered |
| 2 | A\* with weighted cells, no corner cutting, scaled octile heuristic, node counting | Optimal on hand-checked small cases |
| 3 | Compiler + product-graph search; chaining variant for ablation | All operations verified on hand-written intent forms |
| 4 | `verify_path`, checkers (5 types), intent validation | Correct on known-good and known-bad paths |
| 5 | Property-based tests for T1–T5 against `networkx` reference | Tests pass on thousands of random cases |

**Checkpoint 1:** rendered maps with paths; same map with and without a forbidden region
showing the reroute; preferred-corridor example showing take-it/abandon-it behaviour;
property-test results. *Talking point:* the planning half is complete and empirically
checked against the theorems before any language model is involved.

### Week 2 — Proofs, benchmark, language stage, baselines

| Day | Task | Done when |
|---|---|---|
| 6 | Write `docs/soundness.md` with full proofs of T1–T6 | Reviewed by both of us; every assumption maps to a code invariant |
| 7 | Instruction benchmark: templates, hand-written set, paraphrases, gold intent, gold LTL, infeasible set | ~495 inputs stored as JSON |
| 8 | Parser (M3): prompt, JSON schema, validation, retries, caching | Parses the full set without crashing |
| 9 | M2: LTL prompt, Spot syntax check, LTLf → DFA, product planner; T6 cross-check test | M2-oracle == M3-oracle cost on all hard instructions |
| 10 | M1 end-to-end baseline; M0; one command runs everything | `scripts/run_all.py` end to end |

**Checkpoint 2:** `soundness.md`; benchmark statistics; each method run on a sample of
instructions; T6 cross-check result.

### Week 3 — Evaluation and write-up

| Day | Task | Done when |
|---|---|---|
| 11 | Full evaluation run, all methods and ablations | Raw results stored |
| 12 | Metrics, CIs, McNemar, tables per category | Results tables |
| 13 | Error analysis, attribute every failure by stage and type | Failure breakdown table |
| 14 | Figures: validity plot, side-by-side paths (M1 through a wall, M2-harden failing a soft case), T5 detour plot | Figure set |
| 15 | Report, README, cleanup | Repository reproduces every number with one command |

**Checkpoint 3:** results across four methods × six categories; validity headline plot;
soft-preference comparison M2 vs M3; failure breakdown by stage.

**If held to 10 days:** merge Days 4–5 (fewer property tests), Days 9–10 (M2 without
the harden/drop oracle variants), Days 12–13, and Days 14–15. Do **not** cut the
proofs or M2 — those are what was asked for.

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| LLM API rate limits or cost | Cache every response keyed by model + prompt hash; commit the cache so reruns are free and deterministic |
| Parser returns malformed JSON | Structured output mode plus schema validation plus retries |
| Spot install | macOS Apple-Silicon builds exist on conda-forge; use a conda (Miniforge) env with Python 3.11. Fallback: restrict M2 output to the grammar of T6 and hand-write a parser → DFA |
| Free-tier LLM limits (Gemini Flash: per-minute and per-day caps, set per project) | Cache every call; client-side rate limiter with backoff on HTTP 429; spread the full run over 2–3 days; start Day 11 early. Pin the exact model version string in config and in the report |
| Free-tier provider may change limits or models | Keep `nlpath/llm.py` provider-agnostic; fallback provider: a local model via Ollama (report it as a separate, weaker setting, not mixed in) |
| M1 is *too* bad to be interesting | Also report it on 15 × 15 grids |
| M2 ties M3 on hard categories | Expected (T6). Frame the contribution around soft preferences, readability, and the guarantee structure |
| Proofs take longer than Day 6 | Proof sketches are already in §3.3; Day 6 is formalising, not discovering |
| Benchmark takes longer than Day 7 | Templates first; trim hand-written examples |

---

## 7. Deliverables

- Source code with a single reproduction command.
- `docs/soundness.md`: formal model, theorems T1–T6, proofs, scope of the guarantee.
- The benchmark: 20 maps, ~495 instruction inputs with gold intent and gold LTL,
  checker suite.
- Results tables, plots, annotated path figures, failure breakdown.
- Written report: method, guarantees, results, failure analysis, limitations.

---

## 8. Questions to Have Ready for the Professor

**"What exactly is your soundness guarantee?"** Physical validity is unconditional —
no language-model error can produce an invalid path (T1). Given a correct parse, the
path satisfies every hard constraint, is optimal, `NO_PATH` means genuinely infeasible,
and soft preferences can at most multiply path length by 6 (T2–T5). We do not claim
the parse is correct; we measure that rate, and the LTL route carries the same
assumption.

**"Isn't this just a costmap?"** Partly, and say so — cost scaling for soft
preferences is a costmap operation. The contribution is the combination of hard
exclusion, ordered subgoals and graded preferences driven from one sentence, with
guarantees, plus the measurement of what survives the language stage. Ordered
waypoints cannot be expressed in a costmap at all.

**"Why not just use LTL?"** Because of soft preferences. "Stay in the corridor if you
can" is a matter of degree; LTL can only encode it by inventing a threshold the user
never gave, and a fractional threshold is not even expressible in plain LTL. Cost
scaling represents it directly, with a proven bound (T5). On the hard fragment we
match LTL exactly (T6) — and we measure it (M2).

---

## 9. Open Decisions

- **D1 — ordering semantics: SETTLED, strict.** B is not entered at all before A has
  been visited; the goal counts as the last element of the sequence. Planner, the
  `visits_before` checker and gold LTL all use this (§3.1, §3.2, T6).
- **D2 — LLM: SETTLED, Google Gemini API free tier, `gemini-3.5-flash-lite`** (v2.2:
  `gemini-2.5-flash` is closed to new keys and `gemini-3.8-flash` took 70–230 s per call
  on the free tier; Flash-Lite parsed the test prompts correctly in ~1.5 s). Same model,
  temperature 0, for M1–M3. Free-tier inputs may be used by Google for training; this
  is acceptable since the benchmark is synthetic. Check the exact model name and
  current limits in Google AI Studio before Day 8.
- **Environment:** single macOS machine; conda (Miniforge) env, Python 3.11.
