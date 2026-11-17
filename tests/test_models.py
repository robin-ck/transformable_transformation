import re

import pytest

from splat_studio import models


def test_all_models_have_allowed_licence():
    allowed = models.ALLOWED_LICENCES | models.RESEARCH_ONLY_LICENCES
    for spec in models.ALL:
        assert spec.licence in allowed, spec.repo_id


def test_research_only_models_are_never_default_or_outside_editors():
    in_editors = {m for e in models.EDITORS.values() for m in e.models}
    for spec in models.ALL:
        if spec.licence in models.RESEARCH_ONLY_LICENCES:
            assert spec in in_editors, spec.repo_id
    assert not models.editor_spec(None).research_only
    for editor in models.EDITORS.values():
        assert editor.research_only == ("research use only" in editor.name), editor.key


def test_revisions_are_pinned_commits():
    for spec in models.ALL:
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.repo_id


def test_label_is_short_and_traceable():
    assert models.IMAGE_TO_3D.label == "microsoft/TRELLIS-image-large@25e0d31"


def test_repo_ids_unique():
    ids = [spec.repo_id for spec in models.ALL]
    assert len(ids) == len(set(ids))


def test_editor_registry_has_default_with_allowed_licences():
    spec = models.editor_spec(None)
    assert spec.key == models.DEFAULT_EDITOR == "sdxl-depth"
    assert models.editor_spec("sdxl-depth") is spec
    assert spec.licence == "openrail++"
    allowed = models.ALLOWED_LICENCES | models.RESEARCH_ONLY_LICENCES
    for editor in models.EDITORS.values():
        assert editor.licence
        assert all(m.licence in allowed for m in editor.models), editor.key


def test_editor_label_matches_what_previews_stored():
    label = f"{models.PREVIEW_BASE.label} + {models.PREVIEW_DEPTH.label}"
    assert models.editor_spec("sdxl-depth").label == label


def test_unknown_editor_is_refused():
    with pytest.raises(KeyError, match="sdxl-depth"):
        models.editor_spec("qwen-edit")


def test_make_editor_builds_without_loading_models():
    from splat_studio.editing import restyle

    editor = restyle.make_editor()
    assert (editor.key, editor.uses_depth, editor.pipe) == ("sdxl-depth", True, None)
    assert editor.label == models.editor_spec(None).label
    with pytest.raises(KeyError):
        restyle.make_editor("nope")


def test_kontext_editor_runs_in_its_own_class_without_depth():
    from splat_studio.editing import restyle

    spec = models.editor_spec("flux-kontext")
    assert (spec.modal_class, spec.uses_depth, spec.research_only) == (
        "KontextRenderer",
        False,
        True,
    )
    editor = restyle.make_editor("flux-kontext")
    assert (editor.key, editor.uses_depth, editor.pipe) == ("flux-kontext", False, None)
    assert editor.label == "black-forest-labs/FLUX.1-Kontext-dev@24e9ded"


def test_sdpa_shim_drops_false_gqa_and_refuses_true():
    import types

    from splat_studio.editing import restyle

    calls = []

    def old_sdpa(query, key, value, attn_mask=None, scale=None):  # torch 2.4 signature
        calls.append((query, scale))
        return "out"

    f = types.SimpleNamespace(scaled_dot_product_attention=old_sdpa)
    restyle.allow_sdpa_without_gqa(f)
    restyle.allow_sdpa_without_gqa(f)  # wrapping twice is a no-op
    assert f.scaled_dot_product_attention(query=1, key=2, value=3, scale=0.5,
                                          enable_gqa=False) == "out"  # fmt: skip
    assert calls == [(1, 0.5)]
    with pytest.raises(NotImplementedError):
        f.scaled_dot_product_attention(query=1, key=2, value=3, enable_gqa=True)

    def new_sdpa(query, key, value, enable_gqa=False):
        return "new"

    g = types.SimpleNamespace(scaled_dot_product_attention=new_sdpa)
    restyle.allow_sdpa_without_gqa(g)
    assert g.scaled_dot_product_attention is new_sdpa


def test_every_editor_sets_its_steps():
    assert models.editor_spec("sdxl-depth").steps == 30  # Preview tool keeps 30
    for spec in models.EDITORS.values():
        assert 0 < spec.fast_steps < spec.steps, spec.key
