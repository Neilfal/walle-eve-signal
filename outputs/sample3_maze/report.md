# Mission report: Sample 3 - tiny YAML maze

**Outcome:** `EVE_LOCALIZED`  
**Inferred location of EVE:** (8, 8)  
**Confidence:** 99% (probability that EVE is within one cell of the inferred location)  
**Visually confirmed:** no  
**Inferred EVE status:** dormant  
**90% credible region:** 2 cells

## Recommended next step

Move to within one cell of (8, 8) and scan once to visually confirm EVE (confidence 99%). It is 5 moves away (~5 battery); the current charge (15) is enough. EVE appears dormant: approach slowly and try a wake-up signal.

## Evidence found

| step | source | what it says | vs. final belief |
|---|---|---|---|
| 0 | initial_signal @ (0, 0) | Initial signal (t=0): strength 0.30, bearing 140 deg (+/-30). | - |
| 11 | scan @ (4, 6) | Signal at (4, 6): strength 0.395, bearing 112 deg; EVE not within visual range. | - |
| 14 | scan @ (5, 7) | Signal at (5, 7): strength 0.466, bearing 87 deg; EVE not within visual range. | - |
| 41 | scan @ (9, 4) | Signal at (9, 4): strength 0.400, bearing 232 deg; EVE not within visual range. | - |
| 41 | clue c2 @ (7, 4) | Scorch marks and residue suggest EVE is dormant. | n/a |

## Top hypotheses

| cell | probability |
|---|---|
| (9, 9) | 86.9% |
| (9, 8) | 9.1% |
| (9, 7) | 2.4% |
| (9, 6) | 1.2% |
| (8, 8) | 0.2% |

## Strategy timeline

- after action 0: **start -> explore**. Belief peak (5, 7) (19% within 1 cell), entropy 5.40 bits. Battery 30/30. Next scan point (4, 6): 10 steps, expected information gain 4.04 bits + new-area bonus, cost 12 battery; 18 would remain vs 12 needed to get back to a charger. Runner-up (2, 6) scored 3.87 vs 3.93.
- after action 14: **explore -> return**. No scan point is affordable and still leaves a safe return; turning back.
- after action 26: **return -> recharge**. At a charger with 2/30 battery (7%): topping up before the next excursion.
- after action 27: **recharge -> explore**. Belief peak (9, 7) (88% within 1 cell), entropy 2.04 bits. Battery 30/30. Next scan point (9, 4): 13 steps, expected information gain 1.38 bits + new-area bonus, cost 15 battery; 15 would remain vs 15 needed to get back to a charger. Runner-up (9, 3) scored 1.01 vs 1.18.
- after action 41: **explore -> finish**. Belief is concentrated: 99% probability that EVE is within one cell of (8, 8).

## Resources

- actions: 42 (37 moves, 3 scans, 1 recharges)
- battery spent: 43; left 15/30
- time used: 46/120
- final position: (9, 4) (13 steps from a charger)
- stranded: no
- stopped because: Belief is concentrated: 99% probability that EVE is within one cell of (8, 8).

## Evaluation against ground truth (evaluator only)

- true EVE position: (9, 9); error 1.41 cells; within visual range: True
- true status: dormant (inferred correctly: True)
- spoofed scans received: 0; decoy clues found: none
