"""Trends, and the part of the era confound that can actually be measured.

The corpus's headline pattern is a change in pitch over career time. The
obvious way to check whether it is really a recording-era effect — fit career
time and calendar time together and see which carries the signal — **does not
work**, and it is worth being precise about why.

Career months = calendar time - debut date. Once a talent intercept is in the
model, the career and calendar slopes are exactly collinear: this is the
age-period-cohort identification problem, and the design matrix is provably
rank-deficient. Any two-term fit would return coefficients determined by the
solver's choice of pseudo-inverse rather than by the data.

What *is* identifiable is the non-linear part. Each talent's own linear career
trend is removed first; a period effect common to talents who are at different
career stages then shows up in the residuals. A shared linear drift is absorbed
and stays invisible — that limit is real, and it is encoded in a test below
rather than glossed over.

See reference/statistics.md.
"""

from __future__ import annotations

import math

import pytest

from vvc import analysis


class TestTheilSen:
    def test_recovers_a_clean_slope(self):
        xs = list(range(20))
        ys = [3.0 + 2.0 * x for x in xs]
        slope, intercept = analysis.theil_sen(xs, ys)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(3.0)

    def test_is_not_dragged_by_a_wild_outlier(self):
        """Least squares would be. Contaminated clips are exactly this shape."""
        xs = list(range(20))
        ys = [3.0 + 2.0 * x for x in xs]
        ys[10] = 500.0
        slope, _ = analysis.theil_sen(xs, ys)
        assert slope == pytest.approx(2.0, abs=0.2)

    def test_too_few_points_yield_nan_not_a_slope(self):
        slope, _ = analysis.theil_sen([1.0], [2.0])
        assert math.isnan(slope)


class TestPeriodResiduals:
    @staticmethod
    def _talent(
        debut_month: int, *, n: int = 40, slope: float = -0.5, bump=None, step: int = 1
    ):
        """A talent with a linear career trend, optionally plus a shared shock
        in one calendar year. ``step`` controls sampling density so two talents
        can span the same calendar range with very different clip counts."""
        points = []
        for i in range(n):
            career = i * step
            calendar_month = debut_month + career
            year = 2018 + calendar_month // 12
            value = 350.0 + slope * career
            if bump and year == bump[0]:
                value += bump[1]
            points.append(
                analysis.Point(
                    period=f"{year:04d}-{calendar_month % 12 + 1:02d}",
                    career_months=float(career),
                    value=value,
                )
            )
        return points

    def test_a_shared_shock_in_one_calendar_year_is_detected(self):
        """Talents at different career stages all dipping in the same calendar
        year is the signature of a recording-era change, not of voices."""
        series = {
            name: self._talent(debut, bump=(2020, -30.0))
            for name, debut in (("a", 0), ("b", 9), ("c", 18), ("d", 27))
        }
        residuals = analysis.common_period_residuals(series, period="year")
        assert residuals["2020"]["mean_residual"] < -8.0
        assert residuals["2020"]["n_talents"] >= 3

    def test_a_purely_linear_career_trend_leaves_no_period_signal(self):
        series = {
            name: self._talent(debut)
            for name, debut in (("a", 0), ("b", 9), ("c", 18), ("d", 27))
        }
        residuals = analysis.common_period_residuals(series, period="year")
        assert all(abs(row["mean_residual"]) < 3.0 for row in residuals.values())

    def test_a_shared_linear_calendar_drift_is_absorbed_and_invisible(self):
        """The identification limit, stated as a test rather than a caveat: a
        smooth era drift is indistinguishable from career change here."""
        series = {}
        for name, debut in (("a", 0), ("b", 9), ("c", 18), ("d", 27)):
            points = []
            for step in range(40):
                calendar_month = debut + step
                year = 2018 + calendar_month // 12
                points.append(
                    analysis.Point(
                        period=f"{year:04d}-{calendar_month % 12 + 1:02d}",
                        career_months=float(step),
                        # Driven ONLY by calendar time, not career time at all.
                        value=350.0 - 0.5 * calendar_month,
                    )
                )
            series[name] = points
        residuals = analysis.common_period_residuals(series, period="year")
        assert all(abs(row["mean_residual"]) < 3.0 for row in residuals.values())

    def test_talents_are_weighted_equally_regardless_of_clip_count(self):
        """Otherwise a densely sampled talent becomes the 'common' effect.

        Both talents span the same calendar years; only the sparse one carries
        the shock. If clips were pooled, the dense talent's many clean points
        would drown it out.
        """
        series = {
            "dense": self._talent(0, n=48, step=1),
            "sparse": self._talent(0, n=8, step=6, bump=(2020, -60.0)),
        }
        residuals = analysis.common_period_residuals(series, period="year")
        assert residuals["2020"]["n_talents"] == 2
        assert residuals["2020"]["mean_residual"] < -10.0, (
            "the sparse talent's shock was drowned out by the dense one"
        )


class TestBootstrapCI:
    def test_brackets_the_statistic(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        low, high = analysis.bootstrap_ci(values, seed=0)
        assert low <= 12.5 <= high

    def test_is_deterministic_for_a_seed(self):
        values = [1.0, 5.0, 3.0, 9.0, 2.0, 7.0]
        assert analysis.bootstrap_ci(values, seed=3) == analysis.bootstrap_ci(values, seed=3)

    def test_widens_when_there_is_less_data(self):
        """Unlike min-max, which narrows with less data and so reads backwards."""
        many = [10.0 + (i % 5) for i in range(60)]
        few = many[:6]
        wide = analysis.bootstrap_ci(few, seed=1)
        narrow = analysis.bootstrap_ci(many, seed=1)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_a_single_value_has_no_interval_rather_than_a_zero_width_one(self):
        """Zero width would claim certainty from one observation."""
        low, high = analysis.bootstrap_ci([7.0], seed=0)
        assert math.isnan(low) and math.isnan(high)
