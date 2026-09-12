"""
Generates evidence (not just prose claims) that the synthetic dataset has
the properties the brief asks for: meaningful resident preferences, facility
popularity, time patterns, booking lead times, sparsity, imbalance, noise,
and changing (drifting) behaviour.

Drift is detected empirically from the data itself (first-half vs.
second-half facility mix per resident) rather than read off the generator's
hidden parameters, so the plot is evidence about the DATA, not the code that
made it.

Run: python -m src.eda
Output: docs/figures/*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import load_raw, FACILITY_NAMES

OUT_DIR = "docs/figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def facility_popularity(df):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    counts = df["facility"].value_counts().reindex(FACILITY_NAMES)
    ax.bar(counts.index, counts.values, color="#3b6fa0")
    ax.set_ylabel("bookings")
    ax.set_title(f"Facility popularity is imbalanced ({counts.max()/counts.min():.1f}x between busiest and quietest)")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/facility_popularity.png")
    plt.close(fig)


def resident_engagement_sparsity(df):
    counts = df.groupby("resident_id").size()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(counts, bins=40, color="#3b6fa0")
    ax.set_xlabel("bookings per resident (2-year window)")
    ax.set_ylabel("number of residents")
    ax.set_title(f"Engagement is sparse & skewed (median {int(counts.median())}, max {counts.max()})")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/resident_engagement_sparsity.png")
    plt.close(fig)


def preference_concentration(df):
    """For each resident, what share of their bookings went to their single
    most-used facility? A spike near 1.0 = most residents have a clear
    'regular' facility, which is what makes history predictive."""
    shares = (
        df.groupby("resident_id")["facility"]
        .agg(lambda s: s.value_counts(normalize=True).iloc[0])
    )
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(shares, bins=30, color="#3b6fa0")
    ax.set_xlabel("share of a resident's bookings at their #1 facility")
    ax.set_ylabel("number of residents")
    ax.set_title(f"Preferences are peaked, not uniform (median top-facility share {shares.median():.0%})")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/preference_concentration.png")
    plt.close(fig)


def lead_time_by_facility(df):
    df = df.copy()
    df["lead_hours"] = (df["usage_timestamp"] - df["booking_timestamp"]).dt.total_seconds() / 3600
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    data = [np.log1p(df.loc[df["facility"] == f, "lead_hours"]) for f in FACILITY_NAMES]
    ax.boxplot(data, tick_labels=FACILITY_NAMES, showfliers=False)
    ax.set_ylabel("log(1 + lead time hours)")
    ax.set_title("Booking lead time varies hugely by facility (same-day gym vs. weeks-ahead clubhouse)")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/lead_time_by_facility.png")
    plt.close(fig)


def seasonality(df):
    df = df.copy()
    df["month"] = df["usage_timestamp"].dt.month
    pivot = df.pivot_table(index="month", columns="facility", values="booking_id", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=FACILITY_NAMES)
    shares = pivot.div(pivot.sum(axis=0), axis=1)  # normalize each facility to its own scale
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for f in ["Swimming Pool", "Gym", "Clubhouse", "Multipurpose Hall"]:
        ax.plot(shares.index, shares[f], marker="o", label=f)
    ax.set_xlabel("month")
    ax.set_ylabel("share of that facility's yearly bookings")
    ax.set_title("Seasonality: pool peaks in summer, clubhouse/hall peak around Nov-Dec")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/seasonality.png")
    plt.close(fig)


def behaviour_drift_example(df):
    """Find the resident with the largest empirical first-half vs. second-half
    shift in facility mix (total variation distance), among residents with
    enough bookings to make the comparison meaningful."""
    best = None
    for rid, g in df.groupby("resident_id"):
        g = g.sort_values("booking_timestamp")
        if len(g) < 16:
            continue
        mid = len(g) // 2
        first, second = g.iloc[:mid], g.iloc[mid:]
        p1 = first["facility"].value_counts(normalize=True).reindex(FACILITY_NAMES, fill_value=0)
        p2 = second["facility"].value_counts(normalize=True).reindex(FACILITY_NAMES, fill_value=0)
        tv_dist = 0.5 * (p1 - p2).abs().sum()
        if best is None or tv_dist > best[0]:
            best = (tv_dist, rid, p1, p2)

    tv_dist, rid, p1, p2 = best
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(FACILITY_NAMES))
    width = 0.35
    ax.bar(x - width / 2, p1.values, width, label="first half of history", color="#9fb8cf")
    ax.bar(x + width / 2, p2.values, width, label="second half of history", color="#3b6fa0")
    ax.set_xticks(x)
    ax.set_xticklabels(FACILITY_NAMES, rotation=35, ha="right")
    ax.set_ylabel("share of bookings")
    ax.set_title(f"Behaviour drift example: resident {rid} (total-variation shift {tv_dist:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/behaviour_drift_example.png")
    plt.close(fig)
    return rid, tv_dist


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_raw()
    facility_popularity(df)
    resident_engagement_sparsity(df)
    preference_concentration(df)
    lead_time_by_facility(df)
    seasonality(df)
    rid, tv_dist = behaviour_drift_example(df)
    print(f"Wrote 6 figures to {OUT_DIR}/")
    print(f"Largest empirically-detected behaviour drift: resident {rid} (TV distance {tv_dist:.2f})")


if __name__ == "__main__":
    main()
