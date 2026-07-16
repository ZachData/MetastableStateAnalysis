"""
tests/test_core_particles.py — Tests for core/particles.py (core
foundations item 4: the per-particle-record canonical artifact shape).
"""

import numpy as np
import pytest

from core.particles import (
    ParticleTable, default_population_tag, CHECKPOINT_STEP_SENTINEL,
    KEY_COLUMNS, REQUIRED_VALUE_COLUMNS,
)


class TestDefaultPopulationTag:
    def test_negative_labels_are_unclustered(self):
        labels = np.array([-1, 0, 0, 1, -1])
        pop = default_population_tag(labels)
        assert list(pop) == ["unclustered", "clustered", "clustered", "clustered", "unclustered"]

    def test_dtype_is_unicode_not_object(self):
        pop = default_population_tag(np.array([-1, 0]))
        assert pop.dtype.kind == "U"


class TestFromLayer:
    def test_basic_shape(self):
        t = ParticleTable.from_layer(
            model="gpt2-large", prompt_key="wiki_paragraph", layer=5,
            cluster_labels=[-1, 0, 0, 1],
        )
        assert len(t) == 4
        for col in KEY_COLUMNS + REQUIRED_VALUE_COLUMNS:
            assert col in t.columns

    def test_token_position_is_sequential(self):
        t = ParticleTable.from_layer(
            model="gpt2", prompt_key="p", layer=0, cluster_labels=[0, 0, 1],
        )
        assert list(t.columns["token_position"]) == [0, 1, 2]

    def test_population_derived_from_cluster_label_by_default(self):
        t = ParticleTable.from_layer(
            model="gpt2", prompt_key="p", layer=0, cluster_labels=[-1, 0, 1],
        )
        assert list(t.columns["population"]) == ["unclustered", "clustered", "clustered"]

    def test_non_checkpointed_model_uses_sentinel(self):
        t = ParticleTable.from_layer(
            model="gpt2", prompt_key="p", layer=0, cluster_labels=[0, 1],
        )
        assert (t.columns["checkpoint_step"] == CHECKPOINT_STEP_SENTINEL).all()

    def test_checkpointed_model_records_step(self):
        t = ParticleTable.from_layer(
            model="pythia-1.4b-step1000", prompt_key="p", layer=0,
            cluster_labels=[0, 1], checkpoint_step=1000,
        )
        assert (t.columns["checkpoint_step"] == 1000).all()

    def test_v_projections_default_to_nan(self):
        t = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=0, cluster_labels=[0])
        assert np.isnan(t.columns["v_attractive_proj"]).all()
        assert np.isnan(t.columns["v_repulsive_proj"]).all()

    def test_mismatched_token_str_length_raises(self):
        with pytest.raises(ValueError):
            ParticleTable.from_layer(
                model="gpt2", prompt_key="p", layer=0,
                cluster_labels=[0, 1, 2], token_str=["only", "two"],
            )

    def test_extra_columns_stored_separately(self):
        t = ParticleTable.from_layer(
            model="gpt2", prompt_key="p", layer=0, cluster_labels=[0, 1],
            extra={"probe_score": [0.1, 0.9]},
        )
        assert "probe_score" in t.extra
        assert "probe_score" not in t.columns


class TestConcat:
    def test_stacks_rows(self):
        t1 = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=0, cluster_labels=[0, 1])
        t2 = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=1, cluster_labels=[0, 1])
        merged = ParticleTable.concat([t1, t2])
        assert len(merged) == 4
        assert set(merged.columns["layer"].tolist()) == {0, 1}

    def test_empty_input_gives_empty_table(self):
        assert len(ParticleTable.concat([])) == 0

    def test_tolerates_differing_extra_columns(self):
        t1 = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=0, cluster_labels=[0, 1])
        t2 = ParticleTable.from_layer(
            model="gpt2", prompt_key="p", layer=1, cluster_labels=[0, 1],
            extra={"probe_score": [0.3, 0.7]},
        )
        merged = ParticleTable.concat([t1, t2])
        assert len(merged) == 4
        assert np.isnan(merged.extra["probe_score"][:2]).all()
        np.testing.assert_allclose(merged.extra["probe_score"][2:], [0.3, 0.7])

    def test_raises_on_differing_required_columns(self):
        t1 = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=0, cluster_labels=[0, 1])
        t_minimal = ParticleTable(columns={
            "model": np.array(["gpt2", "gpt2"]),
            "checkpoint_step": np.array([-1, -1]),
            "prompt_key": np.array(["p", "p"]),
            "layer": np.array([1, 1]),
            "token_position": np.array([0, 1]),
            "cluster_label": np.array([0, 1]),
            "population": np.array(["clustered", "clustered"]),
        })
        with pytest.raises(ValueError):
            ParticleTable.concat([t1, t_minimal])


class TestFilter:
    """The population selector (plan item 8), reduced to its primitive."""

    def setup_method(self):
        t1 = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=5, cluster_labels=[-1, 0, 0, 1])
        t2 = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=6, cluster_labels=[0, 0, 1, -1])
        self.merged = ParticleTable.concat([t1, t2])

    def test_filter_by_population(self):
        result = self.merged.filter(population="unclustered")
        assert len(result) == 2

    def test_filter_by_layer(self):
        result = self.merged.filter(layer=5)
        assert len(result) == 4

    def test_filter_combines_conditions(self):
        result = self.merged.filter(layer=5, population="clustered")
        assert len(result) == 3

    def test_filter_on_extra_column(self):
        t = ParticleTable.from_layer(
            model="gpt2", prompt_key="p", layer=0, cluster_labels=[0, 0, -1, -1],
            extra={"probe_score": [0.9, 0.8, 0.1, 0.2]},
        )
        result = t.filter(population="unclustered")
        assert list(result.extra["probe_score"]) == [0.1, 0.2]

    def test_unknown_column_raises(self):
        with pytest.raises(KeyError):
            self.merged.filter(not_a_column=1)


class TestToRecords:
    def test_row_count_and_content(self):
        t = ParticleTable.from_layer(
            model="gpt2-large", prompt_key="p", layer=0,
            cluster_labels=[0, 1], token_str=["The", "cat"],
        )
        recs = t.to_records()
        assert len(recs) == 2
        assert recs[0]["model"] == "gpt2-large"
        assert recs[0]["token_str"] == "The"


class TestSaveLoadRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path):
        t1 = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=0, cluster_labels=[-1, 0, 0, 1])
        t2 = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=1, cluster_labels=[0, 0, 1, -1])
        merged = ParticleTable.concat([t1, t2])

        path = tmp_path / "particle_table.npz"
        merged.save(path)
        loaded = ParticleTable.load(path)

        assert len(loaded) == len(merged)
        assert list(loaded.columns["model"]) == list(merged.columns["model"])
        assert list(loaded.columns["population"]) == list(merged.columns["population"])
        np.testing.assert_array_equal(loaded.columns["cluster_label"], merged.columns["cluster_label"])

    def test_extra_columns_round_trip(self, tmp_path):
        t = ParticleTable.from_layer(
            model="gpt2", prompt_key="p", layer=0, cluster_labels=[0, 1],
            extra={"probe_score": [0.4, 0.6]},
        )
        path = tmp_path / "t.npz"
        t.save(path)
        loaded = ParticleTable.load(path)
        np.testing.assert_allclose(loaded.extra["probe_score"], [0.4, 0.6])

    def test_load_does_not_require_allow_pickle(self, tmp_path):
        """Regression guard: string columns must be a fixed-width unicode
        dtype, not dtype=object — otherwise load() (allow_pickle=False)
        would raise on every saved table."""
        t = ParticleTable.from_layer(model="gpt2", prompt_key="p", layer=0, cluster_labels=[0])
        path = tmp_path / "t.npz"
        t.save(path)
        raw = np.load(path, allow_pickle=False)  # would raise if any column were dtype=object
        assert "model" in raw.files

    def test_save_refuses_dtype_object_column(self, tmp_path):
        bad = ParticleTable(columns={
            "model": np.array(["m", "m"], dtype=object),
            "checkpoint_step": np.array([-1, -1]),
            "prompt_key": np.array(["p", "p"]),
            "layer": np.array([0, 0]),
            "token_position": np.array([0, 1]),
            "cluster_label": np.array([0, 1]),
            "population": np.array(["clustered", "clustered"]),
        })
        with pytest.raises(ValueError):
            bad.save(tmp_path / "bad.npz")
