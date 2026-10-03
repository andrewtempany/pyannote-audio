# Assignment classifier — dataset stats and floors

Pair accuracy over labelled target pairs. `identity` = the clustering's own label; `nc` = plain nearest centroid (argmax cosine). Overlap bins by the pair's overlap fraction.

| split | method | meetings | target pairs | held-back | outliers | labelled % | candidates (mean) | identity acc | nc acc | identity by bin (0 / ≤.25 / ≤.5 / >.5) | nc by bin | n by bin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | vbx | 16 | 21745 | 21745 | 0 | 98.6 | 4.19 | 44.2 | 67.2 | 63.8 / 72.1 / 70.1 / 31.2 | 78.2 / 82.1 / 79.3 / 60.3 | 4428 / 1243 / 2176 / 13590 |
| test | hdbscan | 16 | 28205 | 21745 | 6460 | 97.7 | 3.75 | 72.7 | 72.7 | 84.2 / 86.7 / 82.4 / 61.9 | 84.2 / 86.7 / 82.4 / 61.9 | 6243 / 3184 / 3854 / 14271 |
