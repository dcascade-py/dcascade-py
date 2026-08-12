# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 20:54:50 2026

@author: bfinch
"""

import numpy as np

def Global_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb(
    V_dep_old,
    Qbi_incoming,
    ReachData,
    n,
    p_no_mobilize=0.20,             # Fallback if reach-level columns missing/invalid
    b_range=(-0.03, 0.98),          # Random exponent range [min_b, max_b]
    a_fixed=0.5,                    # Load_mob = a_fixed * Vol_dens**b
    rng=None,                       # Optional numpy Generator
    # --- Recruitment controls ---
    t=None,
    recruitment_schedule=None,      # { timestep: [(reach_id0, vol_m3 or [per-class] ), ...], ... }
    recruitment_mode="proportional",
    recruitment_class_index=None,
    recruitment_vector=None,
    # --- Barrier controls ---
    barrier_schedule=None,          # { timestep: [(reach_id0, r or [r_class...]), ...], ... }
    default_retention=0.0           # fallback r in [0,1] if no schedule hit
):
    """
    Mobilization with reach-specific gating from ReachData:
      - If mobten_prob == 1  -> p_no_mob = 0.90
      - If mobten_prob == 0  -> p_no_mob = 1 - mob_prob   (mob_prob is mobilization prob)
      - Else/missing/NaN     -> fall back to p_no_mobilize (function arg)
    """
    if rng is None:
        rng = np.random.default_rng()

    _matrix_compact = globals().get('matrix_compact', lambda x: x)

    # === Reach dimensions & area (ha)
    Wac    = float(ReachData['Wac'].values[n])
    Length = float(ReachData['Length'].values[n])
    Area   = (Wac * Length) / 10000.0

    # === Combine existing and incoming
    Vol_total = np.vstack((V_dep_old, Qbi_incoming))  # (rows, 1+classes)

    # === Total available volume (m³)
    total_volume = float(np.sum(Vol_total[:, 1:]))

    # === Reach-level gating probability (NO-mobilization)
    # Default to provided arg unless valid reach columns are present.
    p_no_mob_gate = float(p_no_mobilize)

    try:
        if 'mobten_prob' in ReachData.columns:
            mt = ReachData['mobten_prob'].values[n]
            mt = int(mt) if not (mt is None) else 0
            if mt == 1:
                # Tenure/threshold condition: strongly suppress mobilization
                p_no_mob_gate = 0.90
            else:
                # Use mobilization probability from the row, convert to NO-mobilization
                if 'mob_prob' in ReachData.columns:
                    mp = ReachData['mob_prob'].values[n]
                    mp = float(mp)
                    if not np.isnan(mp):
                        mp = max(0.0, min(1.0, mp))   # clamp to [0,1]
                        p_no_mob_gate = 1.0 - mp
    except Exception:
        # On any parsing issue, keep fallback p_no_mob_gate
        pass

    # Guard: non-positive area or empty volume → no mobilization; still allow recruitment
    if Area <= 0 or total_volume <= 0:
        V_mob = np.hstack((Vol_total[:, [0]], np.zeros_like(Vol_total[:, 1:])))
        V_dep = Vol_total.copy()
        V_dep = _apply_recruitment_after_mob(
            V_dep, n, t, recruitment_schedule, recruitment_mode,
            recruitment_class_index, recruitment_vector
        )
        return _matrix_compact(V_mob), _matrix_compact(V_dep)

    # === RNG #1: Mobilization gate (now using p_no_mob_gate derived above)
    if rng.random() < p_no_mob_gate:
        V_mob = np.hstack((Vol_total[:, [0]], np.zeros_like(Vol_total[:, 1:])))
        V_dep = Vol_total.copy()
        V_dep = _apply_recruitment_after_mob(
            V_dep, n, t, recruitment_schedule, recruitment_mode,
            recruitment_class_index, recruitment_vector
        )
        return _matrix_compact(V_mob), _matrix_compact(V_dep)

    # === Wood load density (m³/ha)
    Vol_dens = total_volume / Area

    # === RNG #2: draw exponent b ~ Uniform[b_min, b_max]
    b_min, b_max = b_range
    if b_min > b_max:
        b_min, b_max = b_max, b_min  # swap to be safe
    b = float(rng.uniform(b_min, b_max))

    # === Mobilized load (m³/ha) and convert to volume (m³)
    Load_mob = float(a_fixed * (Vol_dens ** b))
    Vol_mob_total = max(0.0, Load_mob * Area)
    Vol_mob_total = min(Vol_mob_total, total_volume)  # cap at available

    # === Proportional allocation by class
    proportions = Vol_total[:, 1:] / total_volume
    V_mob_matrix = Vol_mob_total * proportions
    V_mob_matrix = np.minimum(V_mob_matrix, Vol_total[:, 1:])  # no overdraft

    # === Outputs before recruitment/barrier
    V_mob = np.hstack((Vol_total[:, [0]], V_mob_matrix))
    V_dep = np.hstack((Vol_total[:, [0]], Vol_total[:, 1:] - V_mob_matrix))

    # === Recruitment AFTER mobilization
    V_dep = _apply_recruitment_after_mob(
        V_dep, n, t, recruitment_schedule, recruitment_mode,
        recruitment_class_index, recruitment_vector
    )

    # === Barrier retention AFTER recruitment
    if 'apply_barrier_retention' in globals() and (barrier_schedule is not None or default_retention != 0.0):
        V_mob, V_dep = apply_barrier_retention(
            V_mob, V_dep, n, t,
            barrier_schedule=barrier_schedule,
            default_retention=default_retention
        )

    return _matrix_compact(V_mob), _matrix_compact(V_dep)


def HiLo_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb(
    V_dep_old,
    Qbi_incoming,
    ReachData,
    n,

    # --- Fallback (used only if reach-level columns missing/invalid) ---
    p_no_mobilize=0.20,             # Probability that nothing mobilizes (fallback)

    # --- Default exponent controls ---
    b_range=(-0.10, 0.85),          # Default random exponent range [min_b, max_b]
    a_fixed=0.5,                    # Load_mob = a_fixed * Vol_dens**b
    rng=None,                       # Optional numpy Generator

    # --- Recruitment controls ---
    t=None,
    recruitment_schedule=None,      # { timestep: [(reach_id0, vol_m3 or [per-class] ), ...], ... }
    recruitment_mode="proportional",
    recruitment_class_index=None,
    recruitment_vector=None,

    # --- Barrier controls ---
    barrier_schedule=None,          # { timestep: [(reach_id0, r or [r_class...]), ...], ... }
    default_retention=0.0,          # fallback r in [0,1] if no schedule hit

    # --- Event-driven b exponent controls ---
    event_schedule=None,            # {12: "hi", 34: "lo"} OR {12: [(3,"hi"), (7,"lo")], ...}
    hi_b_range=(0.56, 1.05),        # range used for "hi" events
    lo_b_range=(-0.12, 0.79),       # range used for "lo" events
):
    """
    Power-law mobilization with optional recruitment, barrier retention,
    timestep-based event control of exponent range (hi/lo/custom),
    AND reach-specific mobilization gating using ReachData:
      - If mobten_prob == 1  -> p_no_mob_gate = 0.90
      - If mobten_prob == 0  -> p_no_mob_gate = 1 - mob_prob
      - Else/missing/NaN     -> fall back to p_no_mobilize (function arg)

    event_schedule:
      - Global per timestep:
          {12: "hi", 34: "lo"}   -> applies to all reaches at t=12/34
      - Per reach per timestep:
          {12: [(3,"hi"), (7,"lo")], 34: [(5,(0.2,0.8))]}
        Values may be:
          "hi" -> use hi_b_range
          "lo" -> use lo_b_range
          (bmin, bmax) -> custom override for that (t, reach)

    Falls back to 'b_range' when no matching entry exists.
    """
    import numpy as _np

    if rng is None:
        rng = _np.random.default_rng()

    _matrix_compact = globals().get('matrix_compact', lambda x: x)

    # --- helper to resolve b range for this (t, n)
    def _resolve_b_range(t, n, default_range, event_schedule, hi_range, lo_range):
        if event_schedule is None or t is None:
            return default_range

        evt = event_schedule.get(int(t))
        if evt is None:
            return default_range

        # Two shapes supported:
        # 1) evt is a string ("hi"/"lo") or a tuple range
        if isinstance(evt, str) or _np.isscalar(evt) or isinstance(evt, tuple) or isinstance(evt, list):
            label_or_range = evt
        else:
            # 2) evt is a list of (reach_id, label_or_range)
            label_or_range = None
            try:
                for rid, val in evt:
                    if int(rid) == int(n):
                        label_or_range = val  # last match wins
            except Exception:
                return default_range

        if label_or_range is None:
            return default_range

        if isinstance(label_or_range, str):
            lab = label_or_range.lower()
            if lab == "hi":
                return hi_range
            if lab == "lo":
                return lo_range
            return default_range

        # tuple/list treated as explicit (bmin, bmax)
        try:
            bmin, bmax = float(label_or_range[0]), float(label_or_range[1])
            return (bmin, bmax)
        except Exception:
            return default_range

    # === Reach dimensions & area (ha)
    Wac    = float(ReachData['Wac'].values[n])
    Length = float(ReachData['Length'].values[n])
    Area   = (Wac * Length) / 10000.0

    # === Combine existing and incoming
    Vol_total = _np.vstack((V_dep_old, Qbi_incoming))  # (rows, 1+classes)

    # === Total available volume (m³)
    total_volume = float(_np.sum(Vol_total[:, 1:]))

    # === Reach-level gating probability (NO-mobilization), with fallback
    p_no_mob_gate = float(p_no_mobilize)
    try:
        if 'mobten_prob' in ReachData.columns:
            mt = ReachData['mobten_prob'].values[n]
            mt = int(mt) if not (mt is None) else 0
            if mt == 1:
                p_no_mob_gate = 0.90
            else:
                if 'mob_prob' in ReachData.columns:
                    mp = ReachData['mob_prob'].values[n]
                    mp = float(mp)
                    if not _np.isnan(mp):
                        mp = max(0.0, min(1.0, mp))
                        p_no_mob_gate = 1.0 - mp
    except Exception:
        pass

    # Guard: non-positive area or empty volume → no mobilization; still allow recruitment
    if Area <= 0 or total_volume <= 0:
        V_mob = _np.hstack((Vol_total[:, [0]], _np.zeros_like(Vol_total[:, 1:])))
        V_dep = Vol_total.copy()
        V_dep = _apply_recruitment_after_mob(
            V_dep, n, t, recruitment_schedule, recruitment_mode,
            recruitment_class_index, recruitment_vector
        )
        return _matrix_compact(V_mob), _matrix_compact(V_dep)

    # === RNG #1: Mobilization gate (UPDATED: reach-specific p_no_mob_gate)
    if rng.random() < p_no_mob_gate:
        V_mob = _np.hstack((Vol_total[:, [0]], _np.zeros_like(Vol_total[:, 1:])))
        V_dep = Vol_total.copy()
        V_dep = _apply_recruitment_after_mob(
            V_dep, n, t, recruitment_schedule, recruitment_mode,
            recruitment_class_index, recruitment_vector
        )
        return _matrix_compact(V_mob), _matrix_compact(V_dep)

    # === Wood load density (m³/ha)
    Vol_dens = total_volume / Area

    # === Choose b-range for this timestep / reach (Hi/Lo/custom/default)
    eff_b_range = _resolve_b_range(t, n, b_range, event_schedule, hi_b_range, lo_b_range)

    # === RNG #2: draw exponent b ~ Uniform[b_min, b_max]
    b_min, b_max = eff_b_range
    if b_min > b_max:
        b_min, b_max = b_max, b_min
    b = float(rng.uniform(b_min, b_max))

    # === Mobilized load (m³/ha) and convert to volume (m³)
    Load_mob = float(a_fixed * (Vol_dens ** b))
    Vol_mob_total = max(0.0, Load_mob * Area)
    Vol_mob_total = min(Vol_mob_total, total_volume)

    # === Proportional allocation by class
    proportions = Vol_total[:, 1:] / total_volume
    V_mob_matrix = Vol_mob_total * proportions
    V_mob_matrix = _np.minimum(V_mob_matrix, Vol_total[:, 1:])

    # === Outputs before recruitment/barrier
    V_mob = _np.hstack((Vol_total[:, [0]], V_mob_matrix))
    V_dep = _np.hstack((Vol_total[:, [0]], Vol_total[:, 1:] - V_mob_matrix))

    # === Recruitment AFTER mobilization (add/remove)
    V_dep = _apply_recruitment_after_mob(
        V_dep, n, t, recruitment_schedule, recruitment_mode,
        recruitment_class_index, recruitment_vector
    )

    # === Barrier retention AFTER recruitment
    if 'apply_barrier_retention' in globals() and (barrier_schedule is not None or default_retention != 0.0):
        V_mob, V_dep = apply_barrier_retention(
            V_mob, V_dep, n, t,
            barrier_schedule=barrier_schedule,
            default_retention=default_retention
        )

    return _matrix_compact(V_mob), _matrix_compact(V_dep)


def WoodClass_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb(
    V_dep_old,
    Qbi_incoming,
    ReachData,
    n,
    p_no_mobilize=0.20,             # Fallback if reach-level columns missing/invalid

    # --- Per-class mobilization controls (can be scalar or length = n_classes) ---
    a_fixed = 0.5,
    b_range = [
        (-0.07, 0.95),   # class 0 (deposit_L)
        (-0.52, 0.23),  # class 1 (deposit_S)
    ],

    rng=None,

    # --- Recruitment controls ---
    t=None,
    recruitment_schedule=None,      # { timestep: [(reach_id0, scalar or [per-class]), ...], ... }
    recruitment_mode="proportional",
    recruitment_class_index=None,
    recruitment_vector=None,

    # --- Barrier controls ---
    barrier_schedule=None,          # { timestep: [(reach_id0, r or [r_class...]), ...], ... }
    default_retention=0.0           # fallback r in [0,1] if no schedule hit
):
    """
    Per-class mobilization equations, BUT driven by the COMBINED (all-classes) wood load density:

      Vol_dens_total = (sum over classes of available volume) / Area

    For each class c:
      b_c ~ Uniform[bmin_c, bmax_c]
      Load_mob_c = a_c * (Vol_dens_total ** b_c)     (m³/ha)
      Vol_mob_c  = Load_mob_c * Area                 (m³)

    Then:
      - cap each Vol_mob_c by available volume in that class
      - if sum(Vol_mob_c) > total available, scale down proportionally
      - allocate within each class across rows proportional to that class’s volume
      - recruitment is applied AFTER mobilization
      - barrier retention is applied AFTER recruitment (if provided)

    Reach-specific gating (NO-mobilization probability) is unchanged:
      - mobten_prob == 1  -> p_no_mob = 0.90
      - mobten_prob == 0  -> p_no_mob = 1 - mob_prob
      - else              -> p_no_mobilize (arg)
    """
    if rng is None:
        rng = np.random.default_rng()

    _matrix_compact = globals().get('matrix_compact', lambda x: x)

    # === Reach dimensions & area (ha)
    Wac    = float(ReachData['Wac'].values[n])
    Length = float(ReachData['Length'].values[n])
    Area   = (Wac * Length) / 10000.0

    # === Combine existing and incoming
    Vol_total = np.vstack((V_dep_old, Qbi_incoming))  # (rows, 1+classes)
    n_classes = Vol_total.shape[1] - 1

    # === Total available volume (m³) (combined)
    total_volume = float(np.sum(Vol_total[:, 1:]))

    # === Reach-level gating probability (NO-mobilization)
    p_no_mob_gate = float(p_no_mobilize)
    try:
        if 'mobten_prob' in ReachData.columns:
            mt = ReachData['mobten_prob'].values[n]
            mt = int(mt) if not (mt is None) else 0
            if mt == 1:
                p_no_mob_gate = 0.90
            else:
                if 'mob_prob' in ReachData.columns:
                    mp = ReachData['mob_prob'].values[n]
                    mp = float(mp)
                    if not np.isnan(mp):
                        mp = max(0.0, min(1.0, mp))
                        p_no_mob_gate = 1.0 - mp
    except Exception:
        pass

    # Guard: non-positive area or empty volume → no mobilization; still allow recruitment
    if Area <= 0 or total_volume <= 0 or n_classes <= 0:
        V_mob = np.hstack((Vol_total[:, [0]], np.zeros_like(Vol_total[:, 1:])))
        V_dep = Vol_total.copy()
        V_dep = _apply_recruitment_after_mob(
            V_dep, n, t, recruitment_schedule, recruitment_mode,
            recruitment_class_index, recruitment_vector
        )
        return _matrix_compact(V_mob), _matrix_compact(V_dep)

    # === RNG #1: Mobilization gate
    if rng.random() < p_no_mob_gate:
        V_mob = np.hstack((Vol_total[:, [0]], np.zeros_like(Vol_total[:, 1:])))
        V_dep = Vol_total.copy()
        V_dep = _apply_recruitment_after_mob(
            V_dep, n, t, recruitment_schedule, recruitment_mode,
            recruitment_class_index, recruitment_vector
        )
        return _matrix_compact(V_mob), _matrix_compact(V_dep)

    # === Combined wood load density (m³/ha) drives BOTH class equations
    Vol_dens_total = total_volume / Area

    # ---- helper: expand scalar -> per-class arrays
    def _as_per_class(x, name):
        if np.isscalar(x):
            return np.full(n_classes, float(x), dtype=float)
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != n_classes:
            raise ValueError(f"{name} must be scalar or length n_classes={n_classes}.")
        return arr

    def _as_per_class_ranges(rngs, name):
        # Accept:
        #   (bmin,bmax) -> replicate
        #   [(bmin,bmax), ...] length n_classes
        if isinstance(rngs, (tuple, list, np.ndarray)) and len(np.asarray(rngs).reshape(-1)) == 2 and np.isscalar(rngs[0]) and np.isscalar(rngs[1]):
            bmin, bmax = float(rngs[0]), float(rngs[1])
            return [(bmin, bmax) for _ in range(n_classes)]
        if isinstance(rngs, (list, tuple)) and len(rngs) == n_classes:
            out = []
            for r in rngs:
                bmin, bmax = float(r[0]), float(r[1])
                out.append((bmin, bmax))
            return out
        raise ValueError(f"{name} must be (bmin,bmax) or a list of length n_classes={n_classes} of (bmin,bmax).")

    a_vec = _as_per_class(a_fixed, "a_fixed")
    b_ranges = _as_per_class_ranges(b_range, "b_range")

    # === Available volume per class (combined across rows)
    class_totals = np.sum(Vol_total[:, 1:], axis=0).astype(float)  # shape (n_classes,)

    # === Draw per-class b and compute per-class mobilized volumes (based on Vol_dens_total)
    Vol_mob_by_class = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        bmin, bmax = b_ranges[c]
        if bmin > bmax:
            bmin, bmax = bmax, bmin
        b_c = float(rng.uniform(bmin, bmax))
        Load_mob_c = float(a_vec[c] * (Vol_dens_total ** b_c))   # m³/ha
        Vol_mob_c  = max(0.0, Load_mob_c * Area)                 # m³
        # cap by what's available in that class
        Vol_mob_by_class[c] = min(Vol_mob_c, class_totals[c])

    # === If class equations over-request total available, scale down proportionally
    sum_mob = float(Vol_mob_by_class.sum())
    if sum_mob > total_volume and sum_mob > 0.0:
        Vol_mob_by_class *= (total_volume / sum_mob)

    # === Allocate within each class across rows proportional to that class volume
    V_mob_matrix = np.zeros_like(Vol_total[:, 1:], dtype=float)  # (rows, n_classes)

    for c in range(n_classes):
        tot_c = float(class_totals[c])
        if tot_c <= 0.0 or Vol_mob_by_class[c] <= 0.0:
            continue
        props_c = Vol_total[:, 1 + c] / tot_c
        V_mob_matrix[:, c] = Vol_mob_by_class[c] * props_c
        # safety cap per cell
        V_mob_matrix[:, c] = np.minimum(V_mob_matrix[:, c], Vol_total[:, 1 + c])

    # === Outputs before recruitment/barrier
    V_mob = np.hstack((Vol_total[:, [0]], V_mob_matrix))
    V_dep = np.hstack((Vol_total[:, [0]], Vol_total[:, 1:] - V_mob_matrix))

    # === Recruitment AFTER mobilization (your updated function supports scalar or vector events)
    V_dep = _apply_recruitment_after_mob(
        V_dep, n, t, recruitment_schedule, recruitment_mode,
        recruitment_class_index, recruitment_vector
    )

    # === Barrier retention AFTER recruitment
    if 'apply_barrier_retention' in globals() and (barrier_schedule is not None or default_retention != 0.0):
        V_mob, V_dep = apply_barrier_retention(
            V_mob, V_dep, n, t,
            barrier_schedule=barrier_schedule,
            default_retention=default_retention
        )

    return _matrix_compact(V_mob), _matrix_compact(V_dep)

def HiLo_WoodClass_fixedAmpPwr_rng_brange_recruit_barriers_uniqueMobProb(
    V_dep_old,
    Qbi_incoming,
    ReachData,
    n,

    # --- Fallback (used only if reach-level columns missing/invalid) ---
    p_no_mobilize=0.20,

    # --- Per-class amplitudes (scalar or length n_classes) ---
    a_fixed=0.5,

    # --- Default per-class b ranges (used when no event schedule hit) ---
    # order: class 0 = deposit_L, class 1 = deposit_S
    b_ranges_default=((0.10, 0.90), (-0.05, 0.80)),

    # --- Hi/Lo per-class b ranges (FOUR total) ---
    hi_b_ranges=((0.49, 1.01), (-0.33, 0.33)),   # (hi_L, hi_S)
    lo_b_ranges=((-0.15, 0.76), (-0.59, 0.06)), # (lo_L, lo_S)

    rng=None,

    # --- Recruitment controls ---
    t=None,
    recruitment_schedule=None,      # {timestep: [(reach_id0, scalar or [per-class]), ...]}
    recruitment_mode="proportional",
    recruitment_class_index=None,
    recruitment_vector=None,

    # --- Barrier controls ---
    barrier_schedule=None,          # {timestep: [(reach_id0, r or [r_class...]), ...]}
    default_retention=0.0,

    # --- Event-driven b controls ---
    # Supports:
    #   {12: "hi", 34: "lo"}   (global)
    #   {12: [(3,"hi"), (7,"lo")], 34: [(5,(bL_range,bS_range))]}
    # Where (bL_range,bS_range) can be:
    #   ((bminL,bmaxL),(bminS,bmaxS))  or  [(bminL,bmaxL),(bminS,bmaxS)]
    event_schedule=None,
):
    """
    Two-class mobilization with:
      1) Reach-specific mobilization gating (mobten_prob / mob_prob logic unchanged)
      2) Hi/Lo (or custom) *per-class* b ranges (4 ranges total for hi/lo)
      3) Both class equations are driven by the *combined* load density:
           Vol_dens_total = (V_L + V_S) / Area
      4) Recruitment AFTER mobilization (supports scalar or vector events)
      5) Barrier retention AFTER recruitment
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    _matrix_compact = globals().get('matrix_compact', lambda x: x)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _as_per_class(x, n_classes, name):
        if np.isscalar(x):
            return np.full(n_classes, float(x), dtype=float)
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != n_classes:
            raise ValueError(f"{name} must be scalar or length n_classes={n_classes}.")
        return arr

    def _as_per_class_ranges(rngs, n_classes, name):
        """
        Accept:
          - (bmin,bmax) replicated across classes
          - [(bmin,bmax), ...] length n_classes
          - ((bmin,bmax),(bmin,bmax)) for 2 classes
        """
        # single pair
        if isinstance(rngs, (tuple, list, np.ndarray)) and len(rngs) == 2 and np.isscalar(rngs[0]) and np.isscalar(rngs[1]):
            bmin, bmax = float(rngs[0]), float(rngs[1])
            return [(bmin, bmax) for _ in range(n_classes)]
        # list-of-pairs
        if isinstance(rngs, (list, tuple)) and len(rngs) == n_classes:
            out = []
            for r in rngs:
                bmin, bmax = float(r[0]), float(r[1])
                out.append((bmin, bmax))
            return out
        raise ValueError(f"{name} must be (bmin,bmax) or list length n_classes of (bmin,bmax).")

    def _resolve_event_label_or_custom(t, n, event_schedule):
        """
        Returns one of:
          - None (no event)
          - "hi" / "lo"
          - custom per-class ranges: [(bminL,bmaxL),(bminS,bmaxS)] for this (t,n)
        """
        if event_schedule is None or t is None:
            return None

        evt = event_schedule.get(int(t))
        if evt is None:
            return None

        # Global shape: "hi"/"lo"/custom
        if isinstance(evt, str):
            return evt.lower()

        # Could be a custom two-class spec directly (global custom)
        # e.g., evt = [ (bminL,bmaxL), (bminS,bmaxS) ] or ((...),(....))
        try:
            # if it's 2 items and each looks like a pair, treat as custom per-class
            if len(evt) == 2 and len(evt[0]) == 2 and len(evt[1]) == 2 and np.isscalar(evt[0][0]) is False:
                pass
        except Exception:
            pass

        # Per-reach list: [(reach_id, label_or_custom), ...]
        label_or_custom = None
        try:
            for rid, val in evt:
                if int(rid) == int(n):
                    label_or_custom = val
        except Exception:
            return None

        if label_or_custom is None:
            return None

        if isinstance(label_or_custom, str):
            return label_or_custom.lower()

        # custom for that reach: either ((bLmin,bLmax),(bSmin,bSmax)) or [(..),(..)]
        try:
            c0 = (float(label_or_custom[0][0]), float(label_or_custom[0][1]))
            c1 = (float(label_or_custom[1][0]), float(label_or_custom[1][1]))
            return [c0, c1]
        except Exception:
            return None

    # ----------------------------
    # Reach geometry
    # ----------------------------
    Wac    = float(ReachData['Wac'].values[n])
    Length = float(ReachData['Length'].values[n])
    Area   = (Wac * Length) / 10000.0

    # Combine existing and incoming
    Vol_total = np.vstack((V_dep_old, Qbi_incoming))  # (rows, 1+classes)
    n_classes = Vol_total.shape[1] - 1
    if n_classes <= 0:
        return _matrix_compact(np.hstack((Vol_total[:, [0]], np.zeros_like(Vol_total[:, 1:])))), _matrix_compact(Vol_total.copy())

    # Total available volume (combined)
    total_volume = float(np.sum(Vol_total[:, 1:]))

    # ----------------------------
    # Reach-level gating probability (NO-mobilization)
    # ----------------------------
    p_no_mob_gate = float(p_no_mobilize)
    try:
        if 'mobten_prob' in ReachData.columns:
            mt = ReachData['mobten_prob'].values[n]
            mt = int(mt) if not (mt is None) else 0
            if mt == 1:
                p_no_mob_gate = 0.90
            else:
                if 'mob_prob' in ReachData.columns:
                    mp = ReachData['mob_prob'].values[n]
                    mp = float(mp)
                    if not np.isnan(mp):
                        mp = max(0.0, min(1.0, mp))
                        p_no_mob_gate = 1.0 - mp
    except Exception:
        pass

    # Guard: non-positive area or empty volume → no mobilization; still allow recruitment
    if Area <= 0 or total_volume <= 0:
        V_mob = np.hstack((Vol_total[:, [0]], np.zeros_like(Vol_total[:, 1:])))
        V_dep = Vol_total.copy()
        V_dep = _apply_recruitment_after_mob(
            V_dep, n, t, recruitment_schedule, recruitment_mode,
            recruitment_class_index, recruitment_vector
        )
        return _matrix_compact(V_mob), _matrix_compact(V_dep)

    # RNG #1: Mobilization gate
    if rng.random() < p_no_mob_gate:
        V_mob = np.hstack((Vol_total[:, [0]], np.zeros_like(Vol_total[:, 1:])))
        V_dep = Vol_total.copy()
        V_dep = _apply_recruitment_after_mob(
            V_dep, n, t, recruitment_schedule, recruitment_mode,
            recruitment_class_index, recruitment_vector
        )
        return _matrix_compact(V_mob), _matrix_compact(V_dep)

    # ----------------------------
    # Combined load density drives both classes
    # ----------------------------
    Vol_dens_total = total_volume / Area

    # Per-class a’s and per-class b ranges (defaults)
    a_vec   = _as_per_class(a_fixed, n_classes, "a_fixed")
    b_def   = _as_per_class_ranges(b_ranges_default, n_classes, "b_ranges_default")
    b_hi    = _as_per_class_ranges(hi_b_ranges, n_classes, "hi_b_ranges")
    b_lo    = _as_per_class_ranges(lo_b_ranges, n_classes, "lo_b_ranges")

    # Resolve which b ranges to use at this timestep/reach
    evt = _resolve_event_label_or_custom(t, n, event_schedule)

    if evt == "hi":
        b_use = b_hi
    elif evt == "lo":
        b_use = b_lo
    elif isinstance(evt, list) and len(evt) == n_classes:
        # custom override for this (t,n)
        b_use = [(float(evt[c][0]), float(evt[c][1])) for c in range(n_classes)]
    else:
        b_use = b_def

    # Available volume per class (sum across rows)
    class_totals = np.sum(Vol_total[:, 1:], axis=0).astype(float)  # (n_classes,)

    # Draw per-class b and compute per-class mobilized volumes
    Vol_mob_by_class = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        bmin, bmax = b_use[c]
        if bmin > bmax:
            bmin, bmax = bmax, bmin
        b_c = float(rng.uniform(bmin, bmax))

        Load_mob_c = float(a_vec[c] * (Vol_dens_total ** b_c))  # m³/ha
        Vol_mob_c  = max(0.0, Load_mob_c * Area)                # m³
        Vol_mob_by_class[c] = min(Vol_mob_c, class_totals[c])   # cap by class available

    # If class equations over-request total available, scale down proportionally
    sum_mob = float(Vol_mob_by_class.sum())
    if sum_mob > total_volume and sum_mob > 0.0:
        Vol_mob_by_class *= (total_volume / sum_mob)

    # Allocate within each class across rows proportional to that class volume
    V_mob_matrix = np.zeros_like(Vol_total[:, 1:], dtype=float)
    for c in range(n_classes):
        tot_c = float(class_totals[c])
        if tot_c <= 0.0 or Vol_mob_by_class[c] <= 0.0:
            continue
        props_c = Vol_total[:, 1 + c] / tot_c
        V_mob_matrix[:, c] = Vol_mob_by_class[c] * props_c
        V_mob_matrix[:, c] = np.minimum(V_mob_matrix[:, c], Vol_total[:, 1 + c])

    # Outputs before recruitment/barrier
    V_mob = np.hstack((Vol_total[:, [0]], V_mob_matrix))
    V_dep = np.hstack((Vol_total[:, [0]], Vol_total[:, 1:] - V_mob_matrix))

    # Recruitment AFTER mobilization
    V_dep = _apply_recruitment_after_mob(
        V_dep, n, t, recruitment_schedule, recruitment_mode,
        recruitment_class_index, recruitment_vector
    )

    # Barrier retention AFTER recruitment
    if 'apply_barrier_retention' in globals() and (barrier_schedule is not None or default_retention != 0.0):
        V_mob, V_dep = apply_barrier_retention(
            V_mob, V_dep, n, t,
            barrier_schedule=barrier_schedule,
            default_retention=default_retention
        )

    return _matrix_compact(V_mob), _matrix_compact(V_dep)



### Recruitment ###

def _apply_recruitment_after_mob(
    V_dep, n, t, recruitment_schedule, recruitment_mode,
    recruitment_class_index, recruitment_vector
):
    import numpy as np

    if recruitment_schedule is None or t is None:
        return V_dep

    events = recruitment_schedule.get(t, [])
    if not events:
        return V_dep

    n_classes = V_dep.shape[1] - 1
    if n_classes <= 0:
        return V_dep

    # ---- Collect the net recruitment targeted to this reach, allowing:
    #   (reach_id, scalar)          -> total volume split by mode (old behavior)
    #   (reach_id, [v0, v1, ...])   -> explicit per-class volumes (new behavior)
    vec_net = np.zeros(n_classes, dtype=float)
    scalar_net = 0.0
    saw_vector = False

    for reach_id, vol in events:
        if int(reach_id) != int(n):
            continue

        # vector form
        if isinstance(vol, (list, tuple, np.ndarray)):
            v = np.asarray(vol, dtype=float).reshape(-1)
            if v.size != n_classes:
                raise ValueError("Recruitment vector length must equal number of classes in V_dep.")
            vec_net += v
            saw_vector = True
        else:
            scalar_net += float(vol)

    # Nothing to do
    if (not saw_vector and scalar_net == 0.0) or (saw_vector and np.allclose(vec_net, 0.0) and scalar_net == 0.0):
        return V_dep

    # ---- If any scalar was provided, convert it to a vector using your existing weight logic
    if scalar_net != 0.0:
        def _weights():
            if recruitment_mode == "vector" and recruitment_vector is not None:
                w = np.asarray(recruitment_vector, dtype=float).reshape(-1)
                if w.size != n_classes:
                    raise ValueError("recruitment_vector length must equal number of classes in V_dep.")
                w = np.clip(w, 0.0, None)
                return (w / w.sum()) if w.sum() > 0 else np.ones(n_classes) / n_classes

            if recruitment_mode == "single" and recruitment_class_index is not None:
                idx = int(recruitment_class_index)
                if idx < 0 or idx >= n_classes:
                    raise ValueError("recruitment_class_index out of range.")
                w = np.zeros(n_classes, dtype=float)
                w[idx] = 1.0
                return w

            if recruitment_mode == "proportional":
                comp = V_dep[:, 1:].sum(axis=0)
                return (comp / comp.sum()) if comp.sum() > 0 else np.ones(n_classes) / n_classes

            return np.ones(n_classes, dtype=float) / n_classes

        vec_net += scalar_net * _weights()

    # ---- Apply per-class net recruitment vec_net (positive add, negative remove)
    # Positive -> append a new row
    if np.any(vec_net > 0):
        add_vec = np.clip(vec_net, 0.0, None)
        if add_vec.sum() > 0:
            new_row = np.hstack(([n], add_vec))
            V_dep = np.vstack((V_dep, new_row))

    # Negative -> remove per class by scaling down uniformly across rows
    if np.any(vec_net < 0):
        remove_vec = np.clip(-vec_net, 0.0, None)
        for c in range(n_classes):
            col = 1 + c
            total_c = float(V_dep[:, col].sum())
            if total_c <= 0.0 or remove_vec[c] <= 0.0:
                continue
            frac = min(1.0, remove_vec[c] / total_c)
            V_dep[:, col] = np.clip(V_dep[:, col] * (1.0 - frac), 0.0, None)

    return V_dep

### Barrier ###

def apply_barrier_retention(
    V_mob,              # array: [id | class...]
    V_dep,              # array: [id | class...]
    n,                  # reach index (0-based)
    t,                  # timestep (int)
    barrier_schedule=None,   # {t: [(reach_id0, retention_float_or_vector), ...], ...}
    default_retention=0.0    # fallback if no schedule hit
):
    """
    Apply a barrier 'retention' at (t, n).
    Retention r in [0,1] = fraction of mobilized wood held back locally.
      r can be:
        - scalar -> same retention for all classes
        - vector length = n_classes -> per-class retention
    Returns:
      V_mob_trans  : mobilized volume that is allowed to transfer downstream
      V_dep_updated: deposit updated with retained portion
    """
    if barrier_schedule is None or t is None:
        r = default_retention
    else:
        events = barrier_schedule.get(t, [])
        r = None
        for (reach_id, ret) in events:
            if int(reach_id) == int(n):
                r = ret  # last match wins if multiple
        if r is None:
            r = default_retention

    n_classes = V_mob.shape[1] - 1
    # Normalize r into a vector in [0,1]
    if np.isscalar(r):
        r_vec = np.clip(np.full(n_classes, float(r), dtype=float), 0.0, 1.0)
    else:
        r_vec = np.asarray(r, dtype=float).reshape(-1)
        if r_vec.size != n_classes:
            raise ValueError("Barrier retention vector must have length equal to number of classes.")
        r_vec = np.clip(r_vec, 0.0, 1.0)

    # Split V_mob into transferable + retained, per class
    V_mob_trans = V_mob.copy()
    V_retained  = V_mob.copy()
    V_mob_trans[:, 1:] *= (1.0 - r_vec)
    V_retained[:,  1:] *= r_vec

    # Add retained portion back into V_dep as a new row
    retained_row = np.hstack(([n], V_retained[:, 1:].sum(axis=0)))
    V_dep_updated = np.vstack((V_dep, retained_row))

    return V_mob_trans, V_dep_updated

