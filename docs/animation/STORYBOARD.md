# NAS search reel storyboard

| Scene | Visual cue | Evidence shown | Gate |
| --- | --- | --- | --- |
| 01 · seed | Synthetic image tiles enter the route | Seed, tensor shape, dataset sizes | Offline deterministic input |
| 02 · search | Mixed-operation edges fan out | Candidate op families and DARTS steps | Continuous architecture space |
| 03 · forward | Batch travels through the model | Execution mode, logits shape, loss when available | Real model path or explicit fallback |
| 04 · derive | Alpha weights collapse into edges | Normal/reduction edge counts and operations | Zero op excluded by genotype utility |
| 05 · compile | Discrete cell graph lights up | Concat nodes and compiled network | Evaluation-shaped architecture |
| 06 · export | Evidence packet closes | JSON genotype, config, and limitations | No benchmark claim from synthetic data |

Exact generated values are stored in `nas_search_sequence.json`.
