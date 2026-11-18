# Features (one row per GPS observation × candidate arc)

Defined in `src/pathsnap/features.py`. None of them uses `.route`, so they are identical at training and inference time.
**Each owner should rewrite the "why" column in their own words after checking feature importance.**

| Feature | Meaning | Why it might help (hypothesis to test) |
|---|---|---|
| `dist` | Perpendicular distance GPS point → arc (m) | The true road is usually the closest |
| `dist_rank`, `dist_minus_min`, `dist_ratio_min` | Distance relative to the point's other candidates | Absolute distance varies with GPS noise; relative distance is more stable |
| `n_cand` | Number of candidates for the point | Dense areas (junctions, parallel roads) are harder |
| `frac_along` | Where the projection falls along the arc (0–1) | Projections at the very end of an arc often mean the next arc is correct |
| `seg_len` | Arc length (m) | Short connector arcs behave differently |
| `speed`, `dt_prev`, `dt_next`, `moving` | Kinematics from neighbouring points | Heading is unreliable when stopped or across gaps |
| `heading_diff`, `heading_cos` | Vehicle heading vs. **directed** arc bearing (NaN when speed < 1 m/s) | ~95 % of roads exist as two opposite arcs; only direction separates them |
| `heading_diff_rev` | Same, ignoring direction | Separates crossing roads from parallel ones |
| `win_mean_dist`, `win_max_dist` | Distance from the 3 previous and 3 next points to the same arc | A single noisy point should not decide; the right arc fits the whole neighbourhood |
| `win_mean_heading_diff` | Heading agreement over the same window | Same idea for direction |
| `in_prev_cands`, `in_next_cands` | Arc is also a candidate of the neighbouring point | Continuity |
| `connected_prev_best`, `connected_next_best` | Arc touches the neighbouring point's closest arc | Continuity through the road network |
| `out_deg_v`, `in_deg_u` | Node degrees at the arc's ends | Junction complexity |
| `has_reverse` | The opposite-direction arc exists | One-way vs. two-way roads |

Missing values (NaN headings) are imputed inside each CV fold with the median plus a missing-indicator column (HistGradientBoosting uses NaN natively).

Next steps for the team: feature importance (permutation importance on CV), ablations (e.g. without window features), and record results here.
