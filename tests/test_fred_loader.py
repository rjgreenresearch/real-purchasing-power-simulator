"""Tests for rpps.fred_loader. All tests run offline and are deterministic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from rpps.fred_loader import (
    CATEGORIES,
    FRED_SERIES,
    BatchResult,
    _parse_observations,
    _read_cached_series,
    _sha256_file,
    build_manifest,
    cache_meta_path,
    cache_path,
    catalog_summary,
    download_all,
    download_series,
    get_api_key,
    get_cache_dir,
    is_cached,
    load_meta,
    load_series,
    series_by_category,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


# ---------------------------------------------------------------------------
# Catalog tests
# ---------------------------------------------------------------------------

class TestCatalog:
    def test_all_series_have_required_metadata(self):
        for sid, s in FRED_SERIES.items():
            assert s.id == sid, f"Catalog key {sid!r} != FredSeries.id {s.id!r}"
            assert s.description, f"{sid} has empty description"
            assert s.frequency in {"M", "Q", "A", "W", "D"}, f"{sid} bad frequency"
            assert 1900 <= s.start_year <= 2030, f"{sid} bad start_year"
            assert s.category in CATEGORIES, f"{sid} bad category"

    def test_principal_series_present(self):
        """The principal series referenced in the paper must all be in the catalog."""
        principals = [
            "CPIAUCNS", "CPIAUCSL", "PPIACO", "GDPDEF",  # prices
            "AHETPI", "COMPRNFB",                          # wages
            "OPHNFB",                                      # productivity
            "PSAVERT", "TDSP",                             # household
            "CSUSHPISA", "MSPUS", "MORTGAGE30US",          # housing
            "GASREGW", "APU000074714", "APU0000703112",    # basket items
        ]
        for sid in principals:
            assert sid in FRED_SERIES, f"Principal series {sid} missing from catalog"

    def test_categories_complete(self):
        """Every catalog entry's category must be in the canonical list."""
        for s in FRED_SERIES.values():
            assert s.category in CATEGORIES

    def test_series_by_category_returns_only_that_category(self):
        for cat in CATEGORIES:
            entries = series_by_category(cat)
            assert all(e.category == cat for e in entries)
            assert len(entries) > 0, f"No entries for category {cat}"

    def test_series_by_category_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown category"):
            series_by_category("not-a-real-category")

    def test_catalog_summary_returns_dataframe(self):
        df = catalog_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(FRED_SERIES)
        assert {"series_id", "category", "frequency", "start_year"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------

class TestParseObservations:
    def test_parses_fixture_correctly(self):
        with (FIXTURES_DIR / "CPIAUCNS_sample.json").open() as f:
            payload = json.load(f)
        series = _parse_observations(payload["observations"])
        assert len(series) == 6
        assert series.index[0] == pd.Timestamp("1947-01-01")
        assert series.index[-1] == pd.Timestamp("1947-06-01")
        assert series.iloc[0] == pytest.approx(21.480)
        assert series.iloc[-1] == pytest.approx(22.080)
        assert series.dtype == "float64"

    def test_handles_missing_value_dot(self):
        obs = [
            {"date": "2020-01-01", "value": "100.0"},
            {"date": "2020-02-01", "value": "."},     # FRED's missing marker
            {"date": "2020-03-01", "value": ""},      # empty string
            {"date": "2020-04-01", "value": "102.5"},
        ]
        s = _parse_observations(obs)
        assert len(s) == 4
        assert s.iloc[0] == pytest.approx(100.0)
        assert pd.isna(s.iloc[1])
        assert pd.isna(s.iloc[2])
        assert s.iloc[3] == pytest.approx(102.5)

    def test_handles_unparseable_value(self):
        obs = [{"date": "2020-01-01", "value": "not-a-number"}]
        s = _parse_observations(obs)
        assert pd.isna(s.iloc[0])

    def test_empty_observations_returns_empty_series(self):
        s = _parse_observations([])
        assert len(s) == 0
        assert s.dtype == "float64"


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_dir_is_created(self, tmp_path):
        d = get_cache_dir(tmp_path / "fred")
        assert d.is_dir()

    def test_cache_path_is_under_cache_dir(self, tmp_path):
        path = cache_path("CPIAUCNS", tmp_path / "fred")
        assert path.parent == (tmp_path / "fred").resolve() or \
               path.parent == (tmp_path / "fred")
        assert path.name == "CPIAUCNS.csv"

    def test_meta_path(self, tmp_path):
        path = cache_meta_path("CPIAUCNS", tmp_path / "fred")
        assert path.name == "CPIAUCNS.meta.json"

    def test_is_cached_false_when_absent(self, tmp_path):
        assert not is_cached("DOES_NOT_EXIST", tmp_path / "fred")

    def test_is_cached_true_when_present(self, tmp_path):
        cache_dir = tmp_path / "fred"
        cache_dir.mkdir()
        (cache_dir / "FAKE.csv").write_text("date,value\n2020-01-01,1.0\n")
        assert is_cached("FAKE", cache_dir)

    def test_read_cached_series_round_trip(self, tmp_path):
        # Write a small CSV
        cache_dir = tmp_path / "fred"
        cache_dir.mkdir()
        path = cache_dir / "TEST.csv"
        original = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-03-01"]),
            name="TEST",
        )
        original.to_csv(path, header=True)
        # Read back
        loaded = _read_cached_series(path)
        pd.testing.assert_series_equal(loaded, original.astype("float64"), check_names=False)

    def test_sha256_is_deterministic(self, tmp_path):
        path = tmp_path / "f.csv"
        path.write_text("hello\n")
        h1 = _sha256_file(path)
        h2 = _sha256_file(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

class TestApiKey:
    def test_explicit_argument_is_used(self):
        assert get_api_key("explicit_key") == "explicit_key"

    def test_environment_variable_is_used(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "env_key_value")
        assert get_api_key(None) == "env_key_value"

    def test_explicit_takes_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "env_key_value")
        assert get_api_key("explicit") == "explicit"

    def test_raises_when_no_key_available(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            get_api_key(None)

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "  spaced_key  ")
        assert get_api_key(None) == "spaced_key"


# ---------------------------------------------------------------------------
# download_series with mocked HTTP
# ---------------------------------------------------------------------------

class TestDownloadSeries:
    def _mock_response(self, monkeypatch, fixture_name):
        """Patch requests.get to return a fixture payload."""
        with (FIXTURES_DIR / fixture_name).open() as f:
            payload = json.load(f)
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        monkeypatch.setattr("rpps.fred_loader.requests.get",
                            lambda *a, **kw: mock_resp)
        return payload

    def test_downloads_and_caches(self, tmp_path, monkeypatch):
        self._mock_response(monkeypatch, "CPIAUCNS_sample.json")
        cache_dir = tmp_path / "fred"
        s = download_series("CPIAUCNS", api_key="test_key", cache_dir=cache_dir)
        assert len(s) == 6
        assert cache_path("CPIAUCNS", cache_dir).is_file()
        assert cache_meta_path("CPIAUCNS", cache_dir).is_file()

    def test_uses_cache_on_second_call(self, tmp_path, monkeypatch):
        self._mock_response(monkeypatch, "CPIAUCNS_sample.json")
        cache_dir = tmp_path / "fred"
        # First call: HTTP
        s1 = download_series("CPIAUCNS", api_key="test_key", cache_dir=cache_dir)

        # Now patch HTTP to fail; second call should still succeed via cache
        def fail(*a, **kw):
            raise AssertionError("Should have used cache, not HTTP")
        monkeypatch.setattr("rpps.fred_loader.requests.get", fail)

        s2 = download_series("CPIAUCNS", api_key="test_key", cache_dir=cache_dir)
        pd.testing.assert_series_equal(s1, s2, check_names=False)

    def test_force_re_downloads(self, tmp_path, monkeypatch):
        self._mock_response(monkeypatch, "CPIAUCNS_sample.json")
        cache_dir = tmp_path / "fred"
        download_series("CPIAUCNS", api_key="test_key", cache_dir=cache_dir)

        # Increase a counter on each subsequent fake call
        calls = {"n": 0}

        def counted(*args, **kw):
            calls["n"] += 1
            mock_resp = MagicMock()
            with (FIXTURES_DIR / "CPIAUCNS_sample.json").open() as f:
                mock_resp.json.return_value = json.load(f)
            mock_resp.raise_for_status.return_value = None
            return mock_resp
        monkeypatch.setattr("rpps.fred_loader.requests.get", counted)

        download_series("CPIAUCNS", api_key="test_key", cache_dir=cache_dir, force=True)
        assert calls["n"] == 1

    def test_metadata_is_written(self, tmp_path, monkeypatch):
        self._mock_response(monkeypatch, "CPIAUCNS_sample.json")
        cache_dir = tmp_path / "fred"
        download_series("CPIAUCNS", api_key="test_key", cache_dir=cache_dir)

        meta = load_meta("CPIAUCNS", cache_dir)
        assert meta["series_id"] == "CPIAUCNS"
        assert meta["count"] == 6
        assert meta["n_observations"] == 6
        assert meta["n_missing"] == 0
        assert meta["first_obs_date"] == "1947-01-01"
        assert meta["last_obs_date"] == "1947-06-01"
        assert "sha256" in meta
        assert "downloaded_at" in meta


# ---------------------------------------------------------------------------
# load_series
# ---------------------------------------------------------------------------

class TestLoadSeries:
    def test_raises_when_not_cached(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not cached"):
            load_series("NOT_THERE", cache_dir=tmp_path / "fred")

    def test_loads_existing_series(self, tmp_path):
        cache_dir = tmp_path / "fred"
        cache_dir.mkdir()
        s = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-03-01"]),
            name="TEST",
        )
        s.to_csv(cache_dir / "TEST.csv", header=True)
        loaded = load_series("TEST", cache_dir=cache_dir)
        assert len(loaded) == 3
        assert loaded.iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# download_all (batch)
# ---------------------------------------------------------------------------

class TestDownloadAll:
    def test_skips_already_cached(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "fred"
        cache_dir.mkdir()
        # Pre-cache a single series
        s = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-01"]), name="CPIAUCNS")
        s.to_csv(cache_dir / "CPIAUCNS.csv", header=True)

        # Patch download to fail loudly
        def fail(*a, **kw):
            raise AssertionError("Should not have called HTTP for cached series")
        monkeypatch.setattr("rpps.fred_loader.requests.get", fail)

        result = download_all(
            api_key="test_key",
            cache_dir=cache_dir,
            series_ids=["CPIAUCNS"],
            rate_limit_sleep=0.0,
        )
        assert "CPIAUCNS" in result.skipped
        assert "CPIAUCNS" not in result.succeeded

    def test_batch_result_str(self):
        r = BatchResult(succeeded=["A"], failed={"B": "err"}, skipped=["C"])
        assert "ok=1" in str(r)
        assert "failed=1" in str(r)
        assert "skipped=1" in str(r)
        assert r.total == 3


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_handles_uncached_series(self, tmp_path):
        cache_dir = tmp_path / "fred"
        cache_dir.mkdir()
        out = tmp_path / "manifest.json"
        m = build_manifest(cache_dir=cache_dir, output_path=out)
        # No series cached -> all should show "uncached"
        assert m["n_series_cached"] == 0
        assert m["n_series_in_catalog"] == len(FRED_SERIES)
        assert all(v.get("status") == "uncached" for v in m["series"].values())
        assert out.is_file()

    def test_manifest_picks_up_cached(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "fred"
        cache_dir.mkdir()

        # Pre-cache one series with metadata
        with (FIXTURES_DIR / "CPIAUCNS_sample.json").open() as f:
            payload = json.load(f)
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        monkeypatch.setattr("rpps.fred_loader.requests.get",
                            lambda *a, **kw: mock_resp)
        download_series("CPIAUCNS", api_key="test_key", cache_dir=cache_dir)

        m = build_manifest(cache_dir=cache_dir, output_path=tmp_path / "m.json")
        assert m["n_series_cached"] == 1
        assert m["series"]["CPIAUCNS"]["count"] == 6
        assert m["series"]["AHETPI"]["status"] == "uncached"


# ---------------------------------------------------------------------------
# CLI: missing-API-key path
# ---------------------------------------------------------------------------

class TestCLIMissingApiKey:
    """When FRED_API_KEY is unset and a download is requested, the CLI must
    exit nonzero with a clean one-line error rather than a Python traceback.
    Regression test for the v0.3.4 UX fix.
    """

    def test_download_all_without_api_key_exits_cleanly(self, tmp_path):
        import os
        import subprocess
        import sys

        # Strip FRED_API_KEY from the child env, then force UTF-8 for portability.
        env = {k: v for k, v in os.environ.items() if k != "FRED_API_KEY"}
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-m", "rpps.fred_loader", "--download-all"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        # Nonzero exit.
        assert result.returncode != 0
        # Clean ERROR: line on stderr, not a traceback.
        assert "ERROR:" in result.stderr
        assert "FRED_API_KEY" in result.stderr
        # No traceback noise.
        assert "Traceback" not in result.stderr

    def test_download_single_without_api_key_exits_cleanly(self, tmp_path):
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "FRED_API_KEY"}
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-m", "rpps.fred_loader", "--download", "CPIAUCNS"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        assert result.returncode != 0
        assert "ERROR:" in result.stderr
        assert "FRED_API_KEY" in result.stderr
        assert "Traceback" not in result.stderr

    def test_catalog_works_without_api_key(self):
        """`--catalog` doesn't need FRED_API_KEY since it just lists what
        the package knows about. It must succeed even with no key set."""
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "FRED_API_KEY"}
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-m", "rpps.fred_loader", "--catalog"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        assert result.returncode == 0, (
            f"Stderr: {result.stderr}"
        )
        # Catalog output should mention some known series id.
        assert "CPIAUCNS" in result.stdout
