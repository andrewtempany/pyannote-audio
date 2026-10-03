# Assignment classifier — run ledger

One row per **test** run. Pair accuracy and ΔDER intervals are 90% paired bootstrap over the 16 test meetings (2000 draws, seed 0).

| run_point | clustering | strategy | model_id | sha256 (8) | run_id | pair acc [CI] | acc by overlap bin (0 / ≤.25 / ≤.5 / >.5) | DER | MD | FA | Conf | der_overlap_assigned | JER | count MAE / bias | ΔDER vs identity [CI] | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vbx-identity | vbx | identity | – | – | 20261003T044356Z-0fa3b41d | 44.2 [40.8, 47.3] | 63.8 / 72.1 / 70.1 / 31.2 | 17.05 | 2925.0 | 1099.1 | 1212.2 | 0.2195 | 0.2219 | 0.250 / +0.250 | +0.00 [+0.00, +0.00] | WS0 reference run (floor); Δacc vs identity +0.0 [+0.0, +0.0] |
| hdbscan-identity | hdbscan | identity | – | – | 20261003T044452Z-b96eddd3 | 72.7 [70.8, 74.7] | 84.2 / 86.7 / 82.4 / 61.9 | 16.86 | 2925.0 | 1099.1 | 1152.7 | 0.2132 | 0.2432 | 0.188 / -0.188 | +0.00 [+0.00, +0.00] | WS0 reference run (floor); Δacc vs identity +0.0 [+0.0, +0.0] |
| vbx-nc_target | vbx | nc_target | – | – | 20261003T050432Z-703683ce | 67.2 [65.5, 69.1] | 78.2 / 82.1 / 79.3 / 60.3 | 16.89 | 2925.0 | 1099.1 | 1164.3 | 0.2172 | 0.2192 | 0.250 / +0.250 | -0.16 [-0.26, -0.06] | WS3 non-learned; Δacc vs identity +22.9 [+20.8, +25.8] |
| vbx-constrained_target | vbx | constrained_target | – | – | 20261003T050555Z-d31bf9f4 | 44.2 [40.8, 47.3] | 63.8 / 72.1 / 70.1 / 31.2 | 17.05 | 2925.0 | 1099.1 | 1212.2 | 0.2195 | 0.2219 | 0.250 / +0.250 | +0.00 [+0.00, +0.00] | WS3 non-learned; Δacc vs identity +0.0 [+0.0, +0.0] |
| hdbscan-nc_target | hdbscan | nc_target | – | – | 20261003T050719Z-5e6664f5 | 72.7 [70.8, 74.7] | 84.2 / 86.7 / 82.4 / 61.9 | 16.86 | 2925.0 | 1099.1 | 1152.7 | 0.2132 | 0.2432 | 0.188 / -0.188 | +0.00 [+0.00, +0.00] | WS3 non-learned; Δacc vs identity +0.0 [+0.0, +0.0] |
| hdbscan-constrained_target | hdbscan | constrained_target | – | – | 20261003T050845Z-a985924b | 53.5 [49.9, 56.8] | 72.2 / 79.6 / 74.4 / 33.8 | 16.94 | 2925.0 | 1099.1 | 1178.6 | 0.2125 | 0.2460 | 0.188 / -0.188 | +0.08 [-0.01, +0.19] | WS3 non-learned; Δacc vs identity -19.2 [-22.0, -17.1] |
