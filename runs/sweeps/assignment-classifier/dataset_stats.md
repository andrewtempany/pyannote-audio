# Assignment classifier — dataset stats and floors

Pair accuracy over labelled target pairs. `identity` = the clustering's own label; `nc` = plain nearest centroid (argmax cosine). Overlap bins by the pair's overlap fraction.

| split | method | meetings | target pairs | held-back | outliers | labelled % | candidates (mean) | identity acc | nc acc | identity by bin (0 / ≤.25 / ≤.5 / >.5) | nc by bin | n by bin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | vbx | 136 | 198328 | 198328 | 0 | 97.9 | 4.15 | 46.8 | 68.9 | 71.0 / 79.8 / 76.4 / 32.6 | 81.2 / 85.7 / 82.9 / 61.8 | 34335 / 10696 / 21297 / 127800 |
| train | hdbscan | 136 | 271502 | 198328 | 73174 | 97.0 | 3.95 | 72.6 | 72.6 | 82.7 / 84.3 / 82.4 / 62.4 | 82.7 / 84.3 / 82.4 / 62.4 | 53303 / 35461 / 40854 / 133658 |
| dev | vbx | 18 | 25123 | 25123 | 0 | 98.4 | 4.06 | 51.4 | 70.5 | 71.7 / 80.0 / 76.8 / 38.2 | 83.9 / 87.7 / 84.0 / 62.4 | 4708 / 1380 / 2917 / 15717 |
| dev | hdbscan | 18 | 35828 | 25123 | 10705 | 98.5 | 4.00 | 75.6 | 75.6 | 86.5 / 89.3 / 84.8 / 63.4 | 86.5 / 89.3 / 84.8 / 63.4 | 7388 / 5044 / 6087 / 16755 |
| test | vbx | 16 | 21745 | 21745 | 0 | 98.6 | 4.19 | 44.2 | 67.2 | 63.8 / 72.1 / 70.1 / 31.2 | 78.2 / 82.1 / 79.3 / 60.3 | 4428 / 1243 / 2176 / 13590 |
| test | hdbscan | 16 | 28205 | 21745 | 6460 | 97.7 | 3.75 | 72.7 | 72.7 | 84.2 / 86.7 / 82.4 / 61.9 | 84.2 / 86.7 / 82.4 / 61.9 | 6243 / 3184 / 3854 / 14271 |
