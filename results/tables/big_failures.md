# Failure breakdown by stage and type

| method | stage | type | runs | share |
|---|---|---|---|---|
| M1 | path | walks through a wall or forbidden cell | 145 | 87.9% |
| M1 | path | invented a path on an infeasible instruction | 15 | 9.1% |
| M1 | interpretation | no well-formed answer after 3 calls | 3 | 1.8% |
| M1 | path | stops short of the goal | 1 | 0.6% |
| M1 | interpretation | said impossible on a feasible instruction | 1 | 0.6% |
| M2 | interpretation | semantically wrong formula | 58 | 11.7% |
| M2 | interpretation | soft preference read as prohibition | 1 | 0.2% |
| M2 | ok | correct | 279 | 56.4% |
| M2 | ok | correct despite non-equivalent formula | 157 | 31.7% |
| M3 | ok | correct | 495 | 100.0% |
| M0 | planning | path ignores a constraint | 107 | 64.8% |
| M0 | ok | correct | 58 | 35.2% |
| M3-oracle | ok | correct | 165 | 100.0% |
| M2-oracle | ok | correct | 165 | 100.0% |
