TWC package v3 fix

Why v2 still failed:
- v2 tried to restore `src/fr3_sim` from current `HEAD`.
- In the current GitHub `main`, the legacy tree is no longer present, so a HEAD-based restore cannot work.
- v3 restores missing legacy paths from the earliest git commit that contains them.

If the repository history is shallow, v3 also attempts `git fetch --unshallow origin` before restoring.
