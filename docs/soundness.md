# Soundness of the Natural-Language Path Planner

**Shashi Bhidodiya · Krishna Kumayu** — formal model, theorems T1–T6 with proofs, and
the exact scope of the guarantee. Expands §3 of `Implementation_Plan.md`. Every
assumption the proofs use is listed in §7 with the code that enforces it and the test
that checks it.

---

## 0. What is guaranteed

```
instruction ──(parser: NOT guaranteed, measured)──▶ intent form ──(T1–T5: guaranteed)──▶ path
```

| Theorem | Statement | Holds |
|---|---|---|
| **T1** | the path is physically walkable, or the answer is `NO_PATH` | always — even if the parser is wrong |
| **T2** | the path satisfies every hard constraint of the intent form | always |
| **T3** | `NO_PATH` only if no path satisfies the intent form | always |
| **T4** | the path is the cheapest one satisfying the intent form | always |
| **T5** | soft preferences make the path at most 6× longer than needed | always |
| **T6** | on the hard constraints, the intent form means the same as an LTL formula | on our maps (§6) |

T2–T5 are about the **parsed** intent form. Whether the form correctly reads the
sentence is not guaranteed; the evaluation measures it. So
P(output is correct on the hard constraints) ≥ P(parse is correct on the hard part),
where a correct output is a satisfying path, or `NO_PATH` when the instruction is
impossible.

---

## 1. Definitions

**Map.** A grid of cells; $F$ is the set of free (non-wall) cells. Each free cell
belongs to at most one region; $R_r$ is the set of cells of region $r$.

**Moves.** From a cell, a robot may step to any of its 8 neighbours. Within a set of
cells $S$, a step $v \to u$ is **legal** if $u \in S$ and, for a diagonal step, both
cells beside the diagonal are also in $S$ (no corner cutting). Step length
$d(v,u)$ is 1 (straight) or $\sqrt2$ (diagonal).

**Path.** $\pi = (v_0, \dots, v_n)$ is *valid in $S$* if every step is legal in $S$.
*Physically valid* means valid in $F$.

**Intent form.** $I$ = goal $g$, forbidden regions $X$, waypoints $w_1, \dots, w_m$,
preferred regions $P$, avoided regions $A$. The **sequence** is
$u_1, \dots, u_{m+1} = w_1, \dots, w_m, g$ (the goal is the last element).
$I$ is **well-formed** if all names are on the map, the sequence has no repeats, no
sequence element is forbidden, and no region is both preferred and avoided.

**Allowed cells.** $S_I$ = free cells that are not in a forbidden region. Forbidden
cells are treated exactly like walls — including for the corner rule, so a path
never even grazes a forbidden cell diagonally (it would "fall in", like a pit).

**First entry.** $e_j(\pi)$ = the first index $i$ with $v_i \in R_{u_j}$.

**Satisfying path.** $\pi$ *satisfies* $I$ from start $s$ if: it is valid in $S_I$,
$v_0 = s$, it ends in the goal region, every sequence element is entered, and
$e_1 < e_2 < \dots < e_{m+1}$. This is **strict ordering**: $u_{j+1}$ is not entered at
all before $u_j$. (Going back into an earlier element later is fine.)

**Admissible start.** $s \in S_I$ and $s$ is not inside $u_2, \dots, u_{m+1}$. If the
start is inside a later element, the robot already broke the order.

**Cost.** Each cell has a weight: 0.5 if preferred, 3 if avoided, else 1. The cost of
a path is $J(\pi) = \sum_i d(v_i, v_{i+1}) \cdot w(v_{i+1})$ — every step pays its length
times the weight of the cell it enters. Its length is $L(\pi) = \sum_i d(v_i, v_{i+1})$.
$I_{hard}$ is $I$ without preferences ($P = A = \emptyset$).

## 2. The planner

A **state** is a pair $(v, k)$: the robot is at cell $v$ and has entered the first $k$
sequence elements in order.

- From $(v, k)$ the robot may make any legal step in $S_I$ to a cell $u$, **except**
  into an element that is not yet due: $u$ may not lie in $u_{k+2}, \dots, u_{m+1}$.
- If $u$ lies in the next element $u_{k+1}$, progress becomes $k+1$; otherwise it
  stays $k$.
- Start state: $(s, 1)$ if $s$ is in $u_1$, else $(s, 0)$. Goal states: $(v, m+1)$
  with $v$ in the goal region. Step cost: $d(v,u) \cdot w(u)$.

If the start is not admissible, return `NO_PATH`. Otherwise run A\* on these states
(with a closed set) and heuristic
$h(v) = w_{min} \cdot$ octile distance from $v$ to the nearest goal cell,
where $w_{min}$ is the smallest weight of any allowed cell. Return the cells of the
state path found, or `NO_PATH`.

The **octile distance** between two cells is the length of the shortest 8-connected
path between them on an empty grid: with $\Delta_1, \Delta_2$ the row and column
differences, it is $\max(\Delta_1,\Delta_2) + (\sqrt2 - 1)\min(\Delta_1,\Delta_2)$.

---

## 3. Three facts used by the proofs

**Fact 1 (the heuristic never overestimates, and is consistent).**
For every step $(v,k) \to (u,k')$ with cost $c$: $h(v) \le c + h(u)$, and $h = 0$ on goals.

*Proof.* Octile distance is a shortest-path distance on an empty grid, so it obeys the
triangle inequality: $\mathrm{oct}(v, t) \le d(v,u) + \mathrm{oct}(u, t)$ for any target $t$.
Take $t$ as the goal cell nearest to $u$ and multiply by $w_{min}$:
$h(v) \le w_{min}\, d(v,u) + h(u)$. Every allowed cell has weight $\ge w_{min}$, so
$w_{min}\, d(v,u) \le d(v,u)\, w(u) = c$. On a goal cell the nearest goal is itself, at
distance 0. ∎

*Why the scaling matters.* Without the factor $w_{min}$, a path through preferred cells
(weight 0.5) can cost less than the octile distance, so $h$ overestimates and A\* can
return a worse path. `tests/test_astar.py` contains a 3×4 example: the unscaled
heuristic returns cost 2.414, the true optimum is 2.207.

**Fact 2 (the counter $k$ is honest).** Along any state path of the planner from an
admissible start, $k$ equals the number of sequence elements entered so far, and they
were entered in order.

*Proof.* Initially true: the start is admissible, so it lies at most in $u_1$, and $k$ is
1 exactly in that case. Each step either enters the next element $u_{k+1}$ (for the
first time, because elements after $k$ have never been entered) — then $k$ grows by 1
and the order is kept — or enters no element beyond $u_k$, because $u_{k+2}, \dots$ are
blocked; then $k$ is unchanged and nothing new was entered. ∎

**Fact 3 (every satisfying path is a planner path).** If $\pi$ satisfies $I$ from $s$,
then $s$ is admissible and, writing next to each cell the number of elements entered
so far, $\pi$ becomes a path of planner states from the start state to a goal state,
with the same cost.

*Proof.* The start is admissible: if $s$ were in a later element $u_j$, then
$e_j = 0 \le e_1$, breaking strict order. Every step of $\pi$ is legal in $S_I$. A step
never enters an element that is not yet due: if it entered $u_{k+2}$ or later while
only $k$ elements had been entered, that element would be entered before $u_{k+1}$,
breaking strict order. The counter increases exactly when the next element is
entered, as in the planner's rule; at the end all $m+1$ elements are entered and the
robot is in the goal region. Costs are the same step by step. ∎

---

## 4. Theorems T1–T5

**T1 — Physical validity (unconditional).** For every well-formed intent form — even
one that misreads the instruction — the planner returns either `NO_PATH` or a
physically valid path starting at $s$.

*Proof.* Every step the planner makes is legal in $S_I$, and $S_I \subseteq F$; a step that
is legal in a smaller set is also legal in $F$ (the same cells are free, and more
cells only make the corner rule easier). ∎

*Why it holds even when the parser is wrong:* the LLM can only change the planner
through the intent form, which can only remove cells (forbidden), change positive
weights, or choose which regions to visit. None of these can add a step through a wall.

**T2 — Soundness.** Every path returned satisfies $I$.

*Proof.* Valid in $S_I$ and starts at $s$: as in T1. It ends at a goal state, so $k = m+1$
and the last cell is in the goal region; by Fact 2 all elements were entered in order. ∎

**T3 — Completeness.** If some path satisfies $I$, the planner does not return
`NO_PATH`. So `NO_PATH` is a proof that the parsed instruction is impossible on this map.

*Proof.* By Fact 3 the start passes the admissibility check and a goal state is
reachable. There are finitely many states (cells × $(m+2)$), and A\* with a closed set
keeps exploring until it reaches a goal or runs out of reachable states. ∎

**T4 — Optimality.** The returned path has the smallest cost $J$ among all paths that
satisfy $I$.

*Proof.* With a consistent heuristic (Fact 1) and positive step costs, A\* returns a
cheapest path to a goal state (standard result; Hart, Nilsson & Raphael 1968;
Russell & Norvig, *AIMA* §3.5). By Fact 3, every satisfying path is a planner path of
the same cost, so none is cheaper. ∎

**T5 — Bounded detour for soft preferences.** Let $L^*$ be the length of a shortest
path satisfying $I_{hard}$. Then the returned path $\hat\pi$ has
$$L(\hat\pi) \le \frac{w_{max}}{w_{min}} L^* \le \frac{3}{0.5} L^* = 6 L^*,$$
and with no preferences it is a shortest path. Preferences never change whether a
path exists.

*Proof.* Preferences only change weights, not which paths satisfy $I$, so $I$ and
$I_{hard}$ have the same satisfying paths; let $\pi_L$ be a shortest one. Every weight lies
between $w_{min}$ and $w_{max}$, so for any path $w_{min} L \le J \le w_{max} L$. By T4,
$w_{min} L(\hat\pi) \le J(\hat\pi) \le J(\pi_L) \le w_{max} L^*$. With no preferences all
weights are 1, so $J = L$ and T4 gives a shortest path. ∎

*Meaning:* this is the formal content of "if you can". A preference can never cause an
unbounded detour; the bound is set by the two multipliers.

---

## 5. Sequential chaining (the M3-chain ablation)

Chaining plans start → $u_1$ → $u_2$ → … → goal as separate A\* searches, each going to
the nearest cell of the next element (with the same moves and blocked elements as the
planner), and joins the pieces.

**Proposition C.** Chaining (a) only returns satisfying paths, (b) finds a path whenever
one exists, but (c) is not always cheapest.

*Proof sketch.* (a) Each piece cannot enter elements that are not yet due, so the order
holds; the last piece ends in the goal. (b) Moves are reversible. When chaining stands
in $u_k$, it can walk back along its own path to the start and then follow any
satisfying path to $u_{k+1}$. Both are allowed at this stage: earlier pieces avoided
a larger blocked set (the blocked elements only shrink as $k$ grows), and the corner
rule uses $S_I$ at every stage, so each reversed step is legal. Hence the next search
succeeds. (c) Each piece fixes where it enters the next element without
looking ahead. `tests/test_product.py::test_chaining_is_suboptimal` has an example:
chaining costs $8 + 2\sqrt2 \approx 10.83$, the optimum is $4 + 4\sqrt2 \approx 9.66$. ∎

In the Day 5 property tests, chaining was strictly worse on about 30% of random
instances that have waypoints.

---

## 6. T6 — Correspondence with LTL on the hard constraints

The formal-logic baseline (M2) expresses the hard constraints as a finite-trace LTL
formula over region names. The trace of a path is the sequence of region labels of its
cells. For the intent form $I$:

$$\psi_{m+1} = \mathbf{F}(g \wedge \mathit{last}), \qquad
\psi_j = \neg(u_{j+1} \vee \dots \vee u_{m+1}) \ \mathbf{U}\ (u_j \wedge \psi_{j+1}),$$
$$\varphi_I = \mathbf{G}\,\neg(x_1 \vee \dots \vee x_{|X|}) \ \wedge\ \psi_1 .$$

Here $\mathbf{U}$ is the *strong* until ($\alpha\,\mathbf{U}\,\beta$: $\beta$ eventually holds, and $\alpha$
holds until then), $\mathbf{F}$ is "eventually", $\mathbf{G}$ is "always", and $\mathit{last}$ holds only at the
final position of the trace ($\mathit{last} \equiv \neg\mathbf{X}[!]\top$, "there is no next position";
in the benchmark files it is written `last` and replaced by `!X[!]1` before Spot
reads the formula).

Reading: never enter a forbidden region; and "no later element until $u_1$, then no
element after $u_2$ until $u_2$, …, and finally end in the goal".
Example — "visit A before B, then the lab, avoid the workshop":
`G !workshop & ((!(B | lab)) U (A & ((!lab) U (B & F(lab & last)))))`.

**Lemma 6.** A trace satisfies $\psi_1$ iff every element is entered, the first entries
are in order $e_1 < \dots < e_{m+1}$, and the trace ends in the goal.

*Proof sketch* (induction from the last element backwards). $\psi_{m+1}$ says exactly
"ends in the goal". Assume the claim for $\psi_{j+1}$ (read from some position). The
formula $\psi_j$ says: before some position where $u_j$ holds and $\psi_{j+1}$ holds from
there, none of $u_{j+1}, \dots, u_{m+1}$ appears. So all of those are first entered after
$u_j$ is, which together with $\psi_{j+1}$ gives the ordering for $j, \dots, m+1$.
Conversely, if the first entries are in order, choose that position to be the first
entry of $u_j$: nothing later has appeared before it, and $\psi_{j+1}$ holds from there. ∎

So $\varphi_I$ says: never enter a forbidden region + all of strict ordering + end in the
goal — everything in "satisfies $I_{hard}$" except one thing a formula over *visited*
cells cannot see: **grazing**. Our planner forbids a diagonal step that passes the
corner of a forbidden cell; the formula only checks which cells are entered.

We call a map **corner-separated** if no *physically* legal diagonal step (all four
cells of its 2×2 block free) passes the corner of a region cell without entering that
region. On such a map grazing a forbidden cell is
impossible anyway, so the formula and the intent form agree exactly.

**T6.** On a corner-separated map, a physically valid path satisfies $I_{hard}$ iff its
trace satisfies $\varphi_I$. Therefore an automaton-based planner for $\varphi_I$ with the same
moves and step costs returns the same optimal cost as our planner.

*Our maps.* Rooms are surrounded by walls with 1-cell doorways and the corridor meets
other cells head-on, so all generated maps are corner-separated. This is checked by
`nlpath.mapgen.is_corner_separated` on all 30 benchmark maps and on random generator
seeds (`tests/test_mapgen.py`).

*Uses.* (a) It generates the gold LTL formula of every benchmark instruction; (b) the
Day 9 cross-check M2-oracle = M3-oracle on hard instructions; (c) it shows the intent
form covers the LTL fragment needed for the hard constraints, and goes beyond it with
soft preferences (T5 has no Boolean equivalent). Our system never builds $\varphi_I$; it
appears only in proofs and evaluation.

---

## 7. From the algorithm to the code

| Assumption (used by) | Enforced in | Checked by |
|---|---|---|
| No corner cutting, forbidden = wall (T1, Fact 3) | `astar.grid_moves` on `compiler` allowed cells | `test_astar.py` corner tests; `test_verify.py` |
| Elements not yet due cannot be entered (Facts 2, 3) | `product.blocked_masks`, `product.plan_product` | `test_product.py`; property tests |
| Counter advances on the next element (Fact 2) | `product.advance` | `test_product.py` |
| Step cost $d \cdot w$, weights 0.5 / 1 / 3 > 0 (T4, T5) | `compiler.compile_intent`, `astar.check_weights` | `test_compiler.py`; `pipeline.plan` recomputes cost |
| $h = w_{min} \cdot$ octile to the nearest goal (Fact 1) | `astar.heuristic_scale`, `astar.octile_field` | `test_properties.py::test_heuristic_is_consistent` |
| A\* with a closed set (T3, T4) | `astar.astar` | networkx cross-checks |
| Well-formed intent form (all) | `intent.validate`, called by `compile_intent` | `test_intent.py` |
| Admissible-start guard (T1–T4) | `product.start_problem` | `test_product.py` start tests |
| At most one region per cell (Fact 2, Lemma 6) | `grid.GridMap` (one integer label per cell) | by construction |
| Corner-separated maps (T6) | `mapgen.is_corner_separated` | `test_mapgen.py::test_benchmark_maps_are_corner_separated`, `::test_generated_maps_are_corner_separated` |

**Runtime verifier.** `verify.verify_path` re-checks walkability, forbidden cells
(including grazing), the start, the goal and strict ordering, without importing the
planner. `pipeline.plan` runs it on every returned path and recomputes the cost; a
failure raises `PathVerificationError` and is never returned quietly.

**Property tests** (`tests/test_properties.py`, 1000 random grids each): T1–T4 against an
independent networkx solver (including all `NO_PATH` cases), the T5 bound, heuristic
consistency, and Proposition C.

---

## 8. Limits of the guarantee

- **The parser is not guaranteed.** T2–T5 hold for the parsed form; parse accuracy is
  measured. The guarantee covers the hard constraints; soft preferences are only
  bounded (T5), not "satisfied". The LTL route (M2) has the same structure; the end-to-end LLM route (M1)
  has no guarantee at all.
- **The grid is the model.** "Walkable" means walkable on the 8-connected grid with no
  corner cutting; the robot's real geometry is out of scope.
- **Strict ordering is a design choice** (decision D1).
- **T6 needs corner-separated maps.** All our maps are, and this is checked by a test.
