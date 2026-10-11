# Mission report: Sample 2 - decoys and spoofed signals

**Outcome:** `EVE_LOCATED`  
**Inferred location of EVE:** (14, 18)  
**Confidence:** 100% (probability that EVE is within one cell of the inferred location)  
**Visually confirmed:** yes  
**Inferred EVE status:** active  
**90% credible region:** 1 cells

## Recommended next step

Go to EVE at (14, 18) and dock/meet. It is 1 moves away (~1 battery); the current charge (14) is enough. EVE appears active: expect it to respond to a greeting ping.

## Evidence found

| step | source | what it says | vs. final belief |
|---|---|---|---|
| 0 | initial_signal @ (1, 1) | Initial signal (t=0): strength 0.08, bearing 136 deg (+/-30). | - |
| 27 | scan @ (13, 15) | Signal at (13, 15): strength 0.518, bearing 144 deg; EVE not within visual range. | - |
| 27 | clue c4 @ (14, 13) | Wheel tracks and a dragged cable head S (bearing ~189 deg). | consistent |
| 34 | scan @ (16, 18) | Signal at (16, 18): strength 0.264, bearing 325 deg; EVE not within visual range. | - |
| 54 | scan @ (10, 10) | Signal at (10, 10): strength 0.339, bearing 126 deg; EVE not within visual range. | - |
| 54 | clue c2 @ (10, 11) | Old beacon log: the signal source is roughly 13.3 cells from here. | consistent |
| 66 | scan @ (13, 18) | EVE seen directly at (14, 18); status: active. | - |

## Top hypotheses

| cell | probability |
|---|---|
| (14, 18) | 100.0% |
| (7, 6) | 0.0% |
| (8, 6) | 0.0% |
| (9, 6) | 0.0% |
| (10, 6) | 0.0% |

## Strategy timeline

- after action 0: **start -> explore**. Belief peak (17, 16) (6% within 1 cell), entropy 7.91 bits. Battery 45/45. Next scan point (13, 15): 26 steps, expected information gain 4.00 bits + new-area bonus, cost 28 battery; 17 would remain vs 10 needed to get back to a charger. Runner-up (12, 13) scored 3.38 vs 3.41.
- after action 27: **explore -> verify**. Belief peak (16, 18) (72% within 1 cell), entropy 3.72 bits. Battery 17/45. Next scan point (16, 18): 6 steps, expected information gain 3.46 bits + new-area bonus, cost 8 battery; 9 would remain vs 4 needed to get back to a charger. Runner-up (15, 18) scored 3.35 vs 3.47.
- after action 34: **verify -> return**. Battery low (9/45, 20%). Turning back: 2 steps to the nearest charger.
- after action 36: **return -> recharge**. At a charger with 7/45 battery (16%): topping up before the next excursion.
- after action 37: **recharge -> explore**. Belief peak (10, 11) (23% within 1 cell), entropy 5.89 bits. Battery 45/45. Next scan point (10, 10): 16 steps, expected information gain 3.41 bits + new-area bonus, cost 18 battery; 27 would remain vs 18 needed to get back to a charger. Runner-up (10, 11) scored 3.01 vs 3.12.
- after action 66: **explore -> finish**. EVE visually confirmed at (14, 18); mission goal reached.

## Resources

- actions: 67 (61 moves, 4 scans, 1 recharges)
- battery spent: 69; left 14/45
- time used: 71/220
- final position: (13, 18) (5 steps from a charger)
- stranded: no
- stopped because: EVE visually confirmed at (14, 18); mission goal reached.

## Evaluation against ground truth (evaluator only)

- true EVE position: (14, 18); error 0.0 cells; within visual range: True
- true status: active (inferred correctly: True)
- spoofed scans received: 1; decoy clues found: ['c2']
