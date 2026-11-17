# Trading rules, in plain English

Every number below comes from `config.yaml` (shown as `key`). The code that applies each
rule is named in brackets. Read this file when you want to know *why* the system recommended
something.

## 1. Data must be complete  [`engine.recommend`, `data.data_status`, `forecast.run_forecast`]
The system makes no recommendation unless all of these hold. The reason is shown instead.
- The chosen date has the F&O bhavcopy, the Nifty 50 close, and the India VIX close.
- There are at least `kronos.context` (512) daily index bars up to that date.
- There are no missing trading days in those bars.
- There are at least 200 days of IV history (front-month ATM IV, or failing that India VIX).

## 2. Forecast  [`forecast.run_forecast`]
- Kronos samples `kronos.n_paths` price paths from the chosen date to expiry, using the last
  `kronos.context` daily bars.
- Each path's distance from the median is then multiplied by `calibration.k` (calibration).
- Every later rule uses these calibrated paths.

## 3. Direction score, from −100 (bearish) to +100 (bullish)  [`scoring.direction_score`]
The score is a weighted sum (`scoring.weights`) of four signals, each scaled to the range −1 to +1:
- **Kronos drift:** (Kronos median − spot) ÷ the market's one-sigma move.
- **OI:** the average of (a) PCR mapped linearly so that `pcr_low` → −1 and `pcr_high` → +1, and
  (b) wall asymmetry. A put wall closer to spot than the call wall counts as bullish.
- **Trend:** the average of close vs its 20-day average, close vs its 50-day average, and the
  20-day change, each divided by its `scoring.trend_scales` entry.
- **Skew:** −(today's 25Δ risk reversal − its 1-year median) ÷ `scoring.skew_scale`. Puts that are
  more expensive than usual count as bearish.

Bands:
- |score| < `neutral_band` → neutral
- |score| < `strong_band` → mild
- otherwise → strong

## 4. Premium edge  [`scoring.premium_edge`]
- **R** = the market's P10–P90 range width ÷ Kronos's P10–P90 range width.
- **IV rank** = today's front-month ATM IV within its 1-year range, from 0 to 100. It always
  uses the front month, whichever expiry you pick. India VIX rank is used instead when there
  are fewer than 200 days of ATM IV history. The report shows which source was used and how
  many days.
- The edge is **good** when R ≥ `edge.r_good` and IV rank ≥ `edge.ivr_good`. Otherwise the
  answer is **No trade**.

## 5. Which strategy  [`strategy.select_structures`]
- **Expiry too close:** with `exits.time_exit_days` calendar days or fewer to expiry → No trade.
- **Strong bearish** → bear call spread.
- **Mild bearish** → iron condor skewed bearish.
- **Mild bullish** → iron condor skewed bullish.
- **Strong bullish** → bull put spread.
- **Neutral:**
  - **Butterfly condition:** Kronos's range ≤ `edge.butterfly_width_ratio` × the market's range.
  - **Naked strangle gate:** all of these must hold:
    - |score| ≤ `neutral_band`
    - R ≥ `edge.r_naked`
    - IV rank ≥ `edge.ivr_naked`
    - no event in `events.yaml` before expiry
    - (in rule 6) both short strikes outside Kronos P5–P95
  - **What gets recommended:**
    - butterfly condition passes → iron butterfly, then naked strangle (if its gate passes),
      then iron condor
    - otherwise, naked gate passes → naked strangle, then iron condor
    - otherwise → iron condor
- A naked straddle is never suggested.

## 6. Strikes  [`strategy.pick_short_strike`, `strategy.build_candidate`]
- **Short strike:** whichever is further from spot of the Kronos quantile and the delta strike
  in `strikes.<structure>`, rounded outward to a listed strike.
- **OI walls:** a wall is a strike whose OI is ≥ `walls.oi_multiple` × the median OI on its
  side and that is among that side's top `walls.top_n` strikes. If a wall lies just beyond
  the short strike, the short moves out to the wall. Exception: it stays put when the wall is
  beyond Kronos P(`walls.max_q`) or below `walls.min_delta` delta. The report then says "wall
  ignored: too far".
- **Iron butterfly:** both shorts at the ATM strike.
- **Liquidity:** every leg needs OI ≥ `liquidity.min_oi`, volume ≥ `liquidity.min_volume`, and a valid IV.

## 7. Wings and size  [`strategy.build_candidate`]
- **Wing widths tried:** `wings.widths`. A width is kept only if credit ≥ `wings.min_credit_ratio` ×
  width (iron butterfly: `butterfly_min_credit_ratio`).
- **Lots, defined-risk trades:** floor(`sizing.defined_risk_pct` × capital ÷ (max loss × lot size)).
- **Lots, naked trades:** floor(`sizing.naked_stress_pct` × capital ÷ (loss at a ±`sizing.naked_gap`
  gap × lot size)).
- **Margin cap:** lots are reduced until the total estimated margin is ≤ `sizing.max_margin_pct` ×
  capital. Margin is approximate. Naked margin is `sizing.naked_margin_per_lot`.
- **Choosing the width:** the width with the highest expected profit across Kronos paths for the
  sized position. On a tie, the narrower width wins.

## 8. Pre-entry check and downgrade  [`risk.pre_entry_check`, `engine.evaluate_structures`]
- **A candidate fails if any of these happens:**
  - the 3-day touch probability of a danger level is > `risk.pre_entry.touch` (naked: `naked_touch`)
  - a ±`gap` move would lose more than `gap_loss_mult` × credit
  - an event falls within `risk.event_days` trading days
- **What happens on failure:**
  - **Naked strangle:** becomes an iron condor at the same short strikes.
  - **Other trades:** the short strikes move one `strike_step` further out (butterfly: wings
    one step wider) and the check runs once more.
  - **Still failing:** the trade is rejected.

## 9. Risk levels on open trades  [`risk.assess`, `risk.level_from_metrics`]
- **Danger level:** the short strike. For an iron butterfly, it is the breakeven.
- **Touch probability:** the higher of the Kronos path touch probability and the IV-based one,
  over `risk.window_days`.
- **AMBER if any of these:**
  - touch > `amber_touch` **and** > `amber_touch_mult` × the entry-day value
  - short |Δ| > max(`amber_delta`, entry Δ + `amber_delta_add`)
  - a ±`amber_gap` gap would lose more than `amber_gap_loss_mult` × credit
  - an event within `event_days`
  - the wall behind your short lost more than `wall_oi_drop` of its OI
  - a **new** wall appears between spot and your short (walls already there at entry, or after
    the last roll, don't count)
  - butterfly: spot has covered more than `butterfly_amber_frac` of the distance to breakeven
- **RED if any of these:**
  - touch > `red_touch` **and** > `red_touch_mult` × the entry-day value
  - |Δ| > max(`red_delta`, entry Δ + `red_delta_add`)
  - loss > `red_loss_mult` × credit
  - butterfly: distance covered > `butterfly_red_frac`
- **Naked trades:** use the absolute thresholds in `risk.naked.*`, without the entry-day comparisons.

## 10. What to do  [`risk.decide_action`] — checked in this order
1. Loss ≥ `exits.hard_stop_mult` × credit → **EXIT** (hard stop).
2. RED → **ADJUST** if rolling the tested side out brings in a net credit, the rolled position
   is not RED, and rolls < `exits.max_rolls`. Otherwise → **EXIT**. Rolling the tested side
   further out usually costs a debit, so RED normally means EXIT.
3. P&L ≥ `exits.profit_target` × credit → **EXIT** (profit target).
4. ≤ `exits.time_exit_days` calendar days left → **EXIT** (time exit).
5. AMBER → **HOLD, prepare**, with the roll that would be used.
6. Otherwise → **HOLD**.

**UNKNOWN:** a position whose data can't be loaded (for example, its chain is missing) is shown as
UNKNOWN with "check manually". The other positions are still checked.

## 11. Rolls  [`risk.propose_roll`, `risk.rolled_baseline`, `journal.apply_roll`]
- **What moves:** the tested side's short and its wing are closed. A new short opens at
  today's strike rule, with a wing of the same width.
- **Credit after a roll:** "credit" in every rule becomes the running net credit (all credits
  received minus all debits paid).
- **Comparison point after a roll:** the tested side's entry-day values (touch probability,
  delta, walls) are reset to the new strikes. The untested side keeps its original values.
