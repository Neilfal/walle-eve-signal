# Mission report: Sample 1 - the quiet corridor

**Outcome:** `EVE_LOCALIZED`  
**Inferred location of EVE:** (13, 12)  
**Confidence:** 97% (probability that EVE is within one cell of the inferred location)  
**Visually confirmed:** no  
**Inferred EVE status:** unknown  
**90% credible region:** 6 cells

## Recommended next step

Move to within one cell of (13, 12) and scan once to visually confirm EVE (confidence 97%). It is 7 moves away (~7 battery); the current charge (18) is enough.

## Evidence found

| step | source | what it says | vs. final belief |
|---|---|---|---|
| 0 | initial_signal @ (1, 1) | Initial signal (t=0): strength 0.16, bearing 94 deg (+/-35). | - |
| 17 | scan @ (12, 6) | Signal at (12, 6): strength 0.449, bearing 154 deg; EVE not within visual range. | - |
| 51 | scan @ (8, 10) | Signal at (8, 10): strength 0.564, bearing 104 deg; EVE not within visual range. | - |
| 51 | clue c2 @ (9, 8) | Old beacon log: the signal source is roughly 6.1 cells from here. | consistent |

## Top hypotheses

| cell | probability |
|---|---|
| (13, 12) | 35.4% |
| (12, 12) | 26.3% |
| (13, 11) | 11.0% |
| (12, 13) | 10.2% |
| (13, 13) | 4.5% |

## Strategy timeline

- after action 0: **start -> explore**. Belief peak (13, 6) (15% within 1 cell), entropy 6.66 bits. Battery 36/36. Next scan point (12, 6): 16 steps, expected information gain 3.71 bits + new-area bonus, cost 18 battery; 18 would remain vs 18 needed to get back to a charger. Runner-up (13, 5) scored 3.36 vs 3.42.
- after action 17: **explore -> return**. No scan point is affordable and still leaves a safe return; turning back.
- after action 33: **return -> recharge**. At a charger with 2/36 battery (6%): topping up before the next excursion.
- after action 34: **recharge -> explore**. Belief peak (13, 12) (87% within 1 cell), entropy 3.56 bits. Battery 36/36. Next scan point (8, 10): 16 steps, expected information gain 1.14 bits + new-area bonus, cost 18 battery; 18 would remain vs 18 needed to get back to a charger. Runner-up (9, 9) scored 0.58 vs 0.85.
- after action 51: **explore -> finish**. Belief is concentrated: 97% probability that EVE is within one cell of (13, 12).

## Resources

- actions: 52 (48 moves, 2 scans, 1 recharges)
- battery spent: 52; left 18/36
- time used: 56/150
- final position: (8, 10) (16 steps from a charger)
- stranded: no
- stopped because: Belief is concentrated: 97% probability that EVE is within one cell of (13, 12).

## Evaluation against ground truth (evaluator only)

- true EVE position: (13, 12); error 0.0 cells; within visual range: True
- true status: low_power (inferred correctly: n/a, no status evidence found)
- spoofed scans received: 0; decoy clues found: none
