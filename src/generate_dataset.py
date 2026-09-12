"""
Synthetic facility-booking dataset generator for one residential community.

Simulates ~2 years of booking history for ~320 residents across 8 shared
facilities. Each resident has a persistent "behavior profile" (facility
preferences, preferred days/hours, booking-lead-time habit, engagement
level) so that history is genuinely predictive of the future -- plus
noise, sparsity, class imbalance, and a behavior-drift event for a subset
of residents, so the problem isn't trivial.

Run: python src/generate_dataset.py
Output: data/bookings.csv
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_RESIDENTS = 320
SIM_START = datetime(2024, 1, 1)
SIM_END = datetime(2025, 12, 31, 23, 59)
MAX_BOOKINGS_PER_RESIDENT = 160

# ---------------------------------------------------------------------------
# Facility catalogue
# ---------------------------------------------------------------------------
# popularity: base community-wide weight
# slots: allowed usage hours (facility "opens" only at these hours)
# lead_time_hours: (min, typical-median, max) lead time people book ahead by
# season_boost: multiplier applied to popularity by month (1-12)
FACILITIES = {
    "Gym":            dict(popularity=1.00, slots=list(range(6, 21)),
                            lead_time_hours=(0.5, 14, 72),
                            season_boost={1: 1.4, 2: 1.2, **{m: 1.0 for m in range(3, 13)}}),
    "Swimming Pool":  dict(popularity=0.85, slots=[6, 7, 8, 16, 17, 18, 19],
                            lead_time_hours=(1, 16, 96),
                            season_boost={m: (1.5 if m in (4, 5, 6) else (0.6 if m in (11, 12, 1) else 1.0)) for m in range(1, 13)}),
    "Badminton Court": dict(popularity=0.75, slots=[17, 18, 19, 20, 21],
                             lead_time_hours=(2, 20, 120),
                             season_boost={m: 1.0 for m in range(1, 13)}),
    "Tennis Court":   dict(popularity=0.45, slots=[6, 7, 17, 18, 19],
                            lead_time_hours=(4, 30, 168),
                            season_boost={m: 1.0 for m in range(1, 13)}),
    "Squash Court":   dict(popularity=0.30, slots=[7, 8, 18, 19, 20],
                            lead_time_hours=(2, 18, 96),
                            season_boost={m: 1.0 for m in range(1, 13)}),
    "Clubhouse":      dict(popularity=0.25, slots=[11, 12, 17, 18, 19, 20],
                            lead_time_hours=(48, 240, 1440),
                            season_boost={m: (1.6 if m in (11, 12) else 1.0) for m in range(1, 13)}),
    "Multipurpose Hall": dict(popularity=0.20, slots=[10, 11, 17, 18, 19],
                               lead_time_hours=(48, 200, 1200),
                               season_boost={m: (1.5 if m in (12, 1) else 1.0) for m in range(1, 13)}),
    "Kids Play Area": dict(popularity=0.55, slots=[9, 10, 16, 17, 18],
                            lead_time_hours=(0.25, 6, 48),
                            season_boost={m: (1.3 if m in (4, 5, 6, 12) else 1.0) for m in range(1, 13)}),
}
FACILITY_NAMES = list(FACILITIES.keys())
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def make_resident_profile(rid, rng):
    has_kids = rng.random() < 0.35
    sporty = rng.random() < 0.30
    wellness = rng.random() < 0.45
    move_in = SIM_START - timedelta(days=int(rng.uniform(0, 365 * 4)))

    # base preference weight per facility, then bias by archetype.
    # Low Dirichlet concentration -> sparse, peaked preferences (most residents
    # have 1-2 "regular" facilities), which is what makes booking history
    # genuinely predictive rather than noise.
    base = rng.dirichlet(np.ones(len(FACILITY_NAMES)) * 0.45)
    weights = dict(zip(FACILITY_NAMES, base))
    if has_kids:
        weights["Kids Play Area"] *= rng.uniform(5, 9)
    if sporty:
        for f in ("Tennis Court", "Badminton Court", "Squash Court"):
            weights[f] *= rng.uniform(3, 6)
    if wellness:
        for f in ("Gym", "Swimming Pool"):
            weights[f] *= rng.uniform(2.5, 4.5)
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    day_archetype = rng.choice(["weekday_regular", "weekend_warrior", "mixed"], p=[0.4, 0.3, 0.3])
    if day_archetype == "weekday_regular":
        day_p = np.array([0.20, 0.19, 0.19, 0.19, 0.15, 0.04, 0.04])
    elif day_archetype == "weekend_warrior":
        day_p = np.array([0.06, 0.06, 0.06, 0.06, 0.10, 0.33, 0.33])
    else:
        day_p = np.ones(7) / 7
    day_p = day_p / day_p.sum()

    hour_archetype = rng.choice(["morning", "evening", "mixed"], p=[0.35, 0.45, 0.20])
    # each resident has one "usual" slot per facility (most people use the gym at
    # the same time most days) rather than picking freely among all open slots
    usual_hour = {
        f: int(rng.choice(FACILITIES[f]["slots"], p=hour_preference_for_slots(hour_archetype, FACILITIES[f]["slots"], rng)))
        for f in FACILITY_NAMES
    }

    lead_archetype = rng.choice(["planner", "last_minute", "typical"], p=[0.25, 0.30, 0.45])
    lead_scale = {"planner": 2.2, "last_minute": 0.35, "typical": 1.0}[lead_archetype]

    # engagement: bookings per week, heavily skewed (many light users, few power users)
    engagement = float(rng.lognormal(mean=-1.1, sigma=0.9))
    engagement = min(engagement, 4.0)

    # 15% of residents undergo a behavior shift partway through the timeline
    drift = rng.random() < 0.15
    drift_date = SIM_START + timedelta(days=int(rng.uniform(200, 650))) if drift else None
    drift_weights = None
    if drift:
        drift_base = rng.dirichlet(np.ones(len(FACILITY_NAMES)) * 0.45)
        drift_weights = dict(zip(FACILITY_NAMES, drift_base))
        if rng.random() < 0.5:
            for f in ("Gym", "Swimming Pool"):
                drift_weights[f] *= rng.uniform(2, 4)
        dt = sum(drift_weights.values())
        drift_weights = {k: v / dt for k, v in drift_weights.items()}

    return dict(
        resident_id=f"R-{rid:04d}", move_in=move_in, household_size=int(rng.integers(1, 6)),
        has_kids=has_kids, sporty=sporty, wellness=wellness,
        facility_weights=weights, day_p=day_p, hour_archetype=hour_archetype,
        usual_hour=usual_hour,
        lead_scale=lead_scale, engagement=max(engagement, 0.03),
        drift_date=drift_date, drift_weights=drift_weights,
    )


def hour_preference_for_slots(hour_archetype, slots, rng):
    slots = np.array(slots)
    if hour_archetype == "morning":
        w = np.where(slots < 12, 3.0, 0.5)
    elif hour_archetype == "evening":
        w = np.where(slots >= 15, 3.0, 0.5)
    else:
        w = np.ones(len(slots))
    w = w + rng.uniform(0, 0.3, size=len(slots))
    return w / w.sum()


def current_weights(profile, t):
    if profile["drift_date"] is not None and t >= profile["drift_date"]:
        days_since = (t - profile["drift_date"]).days
        blend = min(1.0, days_since / 60.0)  # gradual transition over ~2 months
        w = {f: (1 - blend) * profile["facility_weights"][f] + blend * profile["drift_weights"][f]
             for f in FACILITY_NAMES}
    else:
        w = profile["facility_weights"]
    return w


def pick_facility(profile, t, rng):
    w = current_weights(profile, t)
    month = t.month
    combined = np.array([
        w[f] * (0.6 + 0.4 * FACILITIES[f]["popularity"]) * FACILITIES[f]["season_boost"][month]
        for f in FACILITY_NAMES
    ])
    if rng.random() < 0.06:  # noise: occasional random exploration
        combined = np.ones(len(FACILITY_NAMES))
    combined = combined / combined.sum()
    return rng.choice(FACILITY_NAMES, p=combined)


def next_usage_datetime(profile, facility, t, rng):
    day_p = profile["day_p"]
    target_dow = rng.choice(7, p=day_p)
    days_ahead = (target_dow - t.weekday()) % 7
    days_ahead += 7 * rng.integers(0, 2)  # sometimes push a week further out
    if days_ahead == 0 and rng.random() < 0.5:
        days_ahead = 7
    usage_date = t + timedelta(days=int(days_ahead))

    slots = FACILITIES[facility]["slots"]
    if rng.random() < 0.78:
        hour = profile["usual_hour"][facility]
    else:
        hp = hour_preference_for_slots(profile["hour_archetype"], slots, rng)
        hour = int(rng.choice(slots, p=hp))
    usage_dt = usage_date.replace(hour=hour, minute=int(rng.choice([0, 15, 30, 45])), second=0, microsecond=0)
    return usage_dt


def pick_lead_time_hours(profile, facility, rng):
    lo, med, hi = FACILITIES[facility]["lead_time_hours"]
    sigma = 0.7
    mu = np.log(max(med * profile["lead_scale"], 0.3))
    val = rng.lognormal(mean=mu, sigma=sigma)
    val += rng.normal(0, med * 0.05)  # small extra noise
    return float(np.clip(val, lo * 0.5, hi * 1.5))


def simulate_resident(profile, rng):
    rows = []
    t = max(profile["move_in"], SIM_START)
    while t < SIM_END and len(rows) < MAX_BOOKINGS_PER_RESIDENT:
        gap_weeks = rng.exponential(1.0 / max(profile["engagement"], 1e-3))
        t = t + timedelta(days=float(gap_weeks * 7))
        if t >= SIM_END:
            break
        facility = pick_facility(profile, t, rng)
        usage_dt = next_usage_datetime(profile, facility, t, rng)
        lead_hours = pick_lead_time_hours(profile, facility, rng)
        booking_dt = usage_dt - timedelta(hours=lead_hours)
        if booking_dt < profile["move_in"]:
            booking_dt = profile["move_in"]
        if usage_dt > SIM_END:
            continue
        rows.append(dict(
            resident_id=profile["resident_id"],
            facility=facility,
            booking_timestamp=booking_dt,
            usage_timestamp=usage_dt,
            household_size=profile["household_size"],
            has_kids=profile["has_kids"],
            move_in_date=profile["move_in"],
        ))
        t = usage_dt
    return rows


def generate():
    profiles = [make_resident_profile(i + 1, rng) for i in range(N_RESIDENTS)]
    all_rows = []
    for profile in profiles:
        all_rows.extend(simulate_resident(profile, rng))

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["booking_timestamp"]).reset_index(drop=True)

    # introduce a small amount of missing / noisy data for realism
    noise_idx = df.sample(frac=0.01, random_state=RNG_SEED).index
    df.loc[noise_idx, "household_size"] = np.nan

    df.insert(0, "booking_id", [f"B-{i+1:06d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/bookings.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} bookings across {df['resident_id'].nunique()} residents -> {out_path}")
    print(df["facility"].value_counts())
    print("\nDate range:", df["booking_timestamp"].min(), "to", df["booking_timestamp"].max())
