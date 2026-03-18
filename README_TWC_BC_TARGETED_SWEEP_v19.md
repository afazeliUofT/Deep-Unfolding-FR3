# TWC BC targeted sweep v19

This package adds one more **targeted** sweep around the strongest v18 variant (**C**). No core code is changed.

Why v19 is justified:
- v18 resolved the engineering issues.
- Variant **C** was the best learned feasible full-run point, but it still trailed the classical repaired+recovered feasible baseline.
- So the right next step is a narrow **C-neighborhood sweep**, not another algorithm rewrite.

Variants:
- **D**: rate-biased feasible C+ (earlier recovered-teacher switch, slightly more permissive recovery candidate set)
- **E**: teacher-heavy stable C+ (stronger recovered-teacher imitation)
- **F**: latency-oriented C+ (smaller student, shorter repair/recovery stack)

All result roots and job names are unique, so the six slurms can run simultaneously.
