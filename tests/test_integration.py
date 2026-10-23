"""End-to-end: research on train/val, one-time test evaluation, paper-mode replay."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from raygold.config import Mode, Settings
from raygold.data.synthetic import synthetic_m1
from raygold.research.grid import default_exits, default_filters, research_variants
from raygold.research.holdout import HoldoutBurned, evaluate_test
from raygold.research.protocol import SelectionRules, Splits
from raygold.research.run import run_research
from raygold.storage import Store

pytestmark = pytest.mark.slow

SPLITS = Splits("2024-01-01", "2024-03-01", "2024-04-01", "2024-05-01")


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return synthetic_m1(130_000, seed=5, start="2024-01-01 00:00")


@pytest.fixture(scope="module")
def researched(tmp_path_factory, data):  # noqa: ANN001
    tmp = tmp_path_factory.mktemp("research")
    s = Settings()
    s.storage.db_path = str(tmp / "db.sqlite")
    store = Store(s.storage.db_path)
    vs = [v for v in research_variants(1) if v.family in ("sweep_reversal", "fvg_retest", "random_baseline")]
    grid = (vs, default_exits()[:14], default_filters()[:12])
    rules = SelectionRules(min_train_trades=20, min_val_trades=10, min_train_t=-99, min_val_t=-99, min_pf=0.0)
    summary = run_research(data, s, SPLITS, rules, tmp / "reports", store, "synthetic", grid=grid,
                           timeframes=("M1", "M5"))
    return tmp, s, store, summary


def test_research_never_sees_test_period(researched) -> None:  # noqa: ANN001
    tmp, s, store, summary = researched
    assert pd.Timestamp(summary["data"]["end"]) < pd.Timestamp(SPLITS.test_start, tz="UTC")
    tr = pd.read_csv(tmp / "reports" / "candidate_trades_validation.csv")
    if len(tr):
        assert pd.to_datetime(tr["exit_time"]).max() < pd.Timestamp(SPLITS.test_start, tz="UTC")
    assert summary["configs_evaluated"] == 2 * 7 * 14 * 12  # 2 timeframes x (3 sweep + 3 fvg + 1 random) variants
    assert (tmp / "reports" / "selected_strategy.json").exists()
    assert len(store.experiments()) >= 1


def test_test_period_is_burned_after_one_use(researched, data) -> None:  # noqa: ANN001
    tmp, s, store, _ = researched
    res = evaluate_test(data, s, SPLITS, tmp / "reports" / "selected_strategy.json", tmp / "reports", store, "synthetic")
    assert res["reused_test_period"] is False
    with pytest.raises(HoldoutBurned):
        evaluate_test(data, s, SPLITS, tmp / "reports" / "selected_strategy.json", tmp / "reports", store, "synthetic")
    again = evaluate_test(data, s, SPLITS, tmp / "reports" / "selected_strategy.json", tmp / "reports", store, "synthetic",
                          allow_reuse=True)
    assert again["reused_test_period"] is True
    payload = json.loads((tmp / "reports" / "selected_strategy.json").read_text())
    assert payload["test_reused"] is True


def test_paper_replay_records_signals_and_trades(researched, data) -> None:  # noqa: ANN001
    from raygold.app import Scanner
    from raygold.backtest.simulator import Costs
    from raygold.mt5.execution import OrderExecutor
    from raygold.mt5.market_data import ReplayBarSource
    from raygold.research.strategy import StrategySpec
    from raygold.risk.manager import RiskManager
    from raygold.risk.sizing import SymbolSpec

    tmp, _, _, _ = researched
    s = Settings()
    s.mode = Mode.PAPER
    s.storage.db_path = str(tmp / "paper.sqlite")
    s.storage.state_path = str(tmp / "state.json")
    s.risk.halt_file = str(tmp / "HALT")
    s.risk.allowed_sessions = ("asia", "london", "newyork", "off")
    from raygold.backtest.simulator import ExitSpec
    from raygold.research.grid import FilterSpec
    from raygold.signals.events import _v

    s.signal.lookback_bars = 4000
    s.signal.min_confidence = 0
    s.risk.risk_per_trade = 5.0  # synthetic stops are wide; the default GBP1 budget rejects most min-lot trades
    s.broker.leverage = 500
    # a frequently-firing spec so the replay exercises sizing, execution and position management
    spec = StrategySpec(_v("sweep_reversal", k=3), ExitSpec(sl_param=2.0, tp_param=1.0, max_hold=60), FilterSpec())
    store = Store(s.storage.db_path)
    rm = RiskManager(s.risk, SymbolSpec(), s.storage.state_path, 7.0, 0.05, 500)
    src = ReplayBarSource(data, 100_000)
    sc = Scanner(s, src, spec, store, rm, OrderExecutor(Mode.PAPER), SymbolSpec(), 1 / 1.27, Costs())
    for _ in range(600):
        sc.step()
        if not src.advance():
            break
    sigs = store.table("signals")
    assert len(sigs) > 0
    assert set(sigs["decision"]) <= {"executed", "rejected", "signal_only", "refused", "failed"}
    trades = store.table("trades")
    assert (trades["mode"] == "PAPER").all()
    assert len(trades) > 0, sigs[["decision", "reason"]].value_counts()
    assert trades["sl"].notna().all()
    assert (trades["status"] == "closed").sum() > 0
