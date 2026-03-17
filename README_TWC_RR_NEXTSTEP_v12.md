TWC v12: rate-recovered feasibility-restored primal-dual track

Why this step:
- v11 proved the repair stack can drive outage to zero, but the final feasibility-restored outputs collapsed rate.
- The collapse came from the repair path itself, not from the raw budget-aware student.
- v12 therefore keeps the stronger budget-aware student and adds a masked feasible rate-recovery stage after guaranteed repair.

What is new:
- budgeted_primal_dual_pgd_repair_recover
- rate_recovered_primal_dual_unfolded
- masked feasible rate recovery with per-sample bisection
- new configs and Slurm files that run in parallel with existing tracks

Intended scientific test:
- compare raw PD student vs repaired+recovered student
- compare repaired+recovered student vs repaired classical PD baseline
- check whether feasibility stays near zero outage while recovering a meaningful fraction of the raw PD rate

Expected success pattern:
- rate_recovered_primal_dual_unfolded should keep outage near zero
- it should clearly improve rate over feasibility_restored_primal_dual_unfolded from v11
- it should beat budgeted_primal_dual_pgd_repair in rate at similar protection
