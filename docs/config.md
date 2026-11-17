# Configuration reference (`config.yaml`)

Every rule threshold and weight lives in `config.yaml`. Changing any value changes the
config hash shown with every recommendation. Units: "pts" = Nifty index points,
"prob" = probability 0–1, "Δ" = absolute option delta 0–1.

| Key | Meaning | Unit | Default | Safe range |
|---|---|---|---|---|
| `capital` | Trading capital used for sizing | INR | 1000000 | > 0 |
| `risk_free_rate` | Rate for Black-76 | annual | 0.065 | 0.04–0.08 |
| `lot_size_fallback` | Lot size when bhavcopy has none (old format) | units | 65 | verify NSE circular |
| `strike_step` | Strike spacing for rounding | pts | 50 | 50 or 100 |
| `data_dir` | Cache, forecasts, journal DB, logs | path | ~/.nifty-advisor | — |
| `kronos.model_id` / `tokenizer_id` | Hugging Face ids | — | Kronos-base | small/base |
| `kronos.context` | History bars fed to Kronos | days | 512 | 256–512 |
| `kronos.n_paths` | Sampled paths per forecast | count | 200 | 100–500 |
| `kronos.temperature`, `top_p` | Sampling settings | — | 1.0, 1.0 | keep 1.0 unless recalibrating |
| `kronos.device` | Torch device | — | cpu | cpu (only cpu is reproducible) |
| `kronos.base_seed` | Root of the deterministic seed | int | 20261118 | any |
| `calibration.k` | Kronos range multiplier around median | × | 1.0 | 0.5–3.0 (fit in Plan 3) |
| `calibration.window_cycles` | Monthly cycles used to fit k | count | 24 | ≥ 6 |
| `calibration.target_coverage` | Target share of closes inside P10–P90 | prob | 0.80 | 0.80 |
| `scoring.weights.*` | Direction score weights (must sum to 1) | — | .35/.25/.20/.20 | each 0–1 |
| `scoring.pcr_low` / `pcr_high` | PCR mapped to −1 / +1 | ratio | 0.7 / 1.3 | — |
| `scoring.neutral_band` | \|score\| below = neutral | score | 20 | 10–30 |
| `scoring.strong_band` | \|score\| at/above = strong | score | 50 | 40–70 |
| `scoring.oi_window_pct` | Strike window around spot for PCR & walls | fraction | 0.10 | 0.05–0.15 |
| `edge.r_good` | Min implied/Kronos width ratio to trade | ratio | 1.10 | 1.0–1.3 |
| `edge.ivr_good` | Min IV rank to trade | 0–100 | 30 | 20–50 |
| `edge.r_naked` / `ivr_naked` | Naked strangle gate | ratio / 0–100 | 1.30 / 50 | stricter only |
| `edge.butterfly_width_ratio` | Kronos width ≤ this × implied → butterfly | ratio | 0.6 | 0.5–0.7 |
| `strikes.<structure>.q` / `.delta` | Kronos quantile and delta cap for shorts | prob / Δ | see file | — |
| `walls.oi_multiple` | Wall = OI ≥ this × median OI on that side | × | 2.0 | 1.5–3 |
| `walls.top_n` | Wall must be in top N OI strikes on its side | count | 3 | 1–5 |
| `walls.min_delta` | Don't move short to a wall below this Δ | Δ | 0.08 | 0.05–0.10 |
| `walls.max_q` | Don't move short beyond Kronos P(max_q)/P(1−max_q) | prob | 0.025 | 0.01–0.05 |
| `liquidity.min_oi` / `min_volume` | Leg strike must have at least this OI / volume | as in bhavcopy | 50000 / 1 | calibrate |
| `wings.widths` | Wing widths tried | pts | 100–500 | multiples of strike_step |
| `wings.min_credit_ratio` | Credit ≥ ratio × width | ratio | 0.25 | 0.2–0.33 |
| `wings.butterfly_min_credit_ratio` | Same for iron butterfly | ratio | 0.50 | 0.4–0.6 |
| `sizing.defined_risk_pct` | Max loss per defined-risk trade | of capital | 0.02 | 0.01–0.03 |
| `sizing.naked_stress_pct` / `naked_gap` | Naked: loss at gap ≤ pct of capital | fraction | 0.04 / 0.05 | — |
| `sizing.max_margin_pct` | Total margin cap | of capital | 0.50 | 0.3–0.6 |
| `sizing.naked_margin_per_lot` | Approx. naked margin | INR/lot | 200000 | from broker |
| `risk.window_days` | Look-ahead for touch probability | trading days | 3 | 2–5 |
| `risk.event_days` | Event within this → AMBER / pre-entry fail | trading days | 3 | 2–5 |
| `risk.amber_touch` / `red_touch` | Touch-prob thresholds (defined risk) | prob | 0.15 / 0.30 | — |
| `risk.amber_touch_mult` / `red_touch_mult` | …and must exceed this × entry value | × | 1.5 / 2.0 | — |
| `risk.amber_delta` / `red_delta` | Short-leg Δ floors | Δ | 0.25 / 0.30 | — |
| `risk.amber_delta_add` / `red_delta_add` | Δ thresholds relative to entry Δ | Δ | 0.08 / 0.15 | — |
| `risk.amber_gap` / `amber_gap_loss_mult` | AMBER if ±gap loss > mult × credit | fraction / × | 0.02 / 1.0 | — |
| `risk.red_loss_mult` | RED if loss > mult × credit | × | 1.5 | 1.0–2.0 |
| `risk.butterfly_amber_frac` / `red_frac` | Spot covered fraction to breakeven | fraction | 0.5 / 0.8 | — |
| `risk.wall_oi_drop` | Wall OI drop vs entry → AMBER | fraction | 0.30 | 0.2–0.5 |
| `risk.naked.*` | Absolute thresholds for naked trades | prob / Δ | .10/.20/.20/.25 | stricter than defined |
| `risk.pre_entry.*` | Candidate pre-entry check thresholds | — | see file | — |
| `risk.stress_moves` | Spot moves for stress P&L | fraction | ±2/3/5% | — |
| `risk.stress_iv_up` | IV shock for stress P&L | vol (0.03 = 3 pts) | 0.03 | — |
| `exits.profit_target` | Exit at this fraction of max profit | fraction | 0.50 | 0.4–0.6 |
| `exits.hard_stop_mult` | Exit when loss ≥ mult × credit | × | 2.0 | 1.5–3 |
| `exits.time_exit_days` | Exit / no entry with ≤ this many calendar days left | days | 7 | 5–10 |
| `exits.max_rolls` | Max adjustments per trade | count | 2 | 0–3 |

`events.yaml`: list of `{date, name, kind}`; kind ∈ rbi, budget, fomc, election, other.
`holidays.yaml`: `{year: [dates]}` of NSE trading holidays — update every December.
