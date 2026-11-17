import re

import pytest

from splat_studio import models


def test_all_models_have_allowed_licence():
    for spec in models.ALL:
        assert spec.licence in models.ALLOWED_LICENCES, spec.repo_id


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
    for editor in models.EDITORS.values():
        assert editor.licence
        assert all(m.licence in models.ALLOWED_LICENCES for m in editor.models), editor.key


def test_editor_label_matches_what_previews_stored():
    label = f"{models.PREVIEW_BASE.label} + {models.PREVIEW_DEPTH.label}"
    assert models.editor_spec("sdxl-depth").label == label


def test_unknown_editor_is_refused():
    with pytest.raises(KeyError, match="sdxl-depth"):
        models.editor_spec("flux-kontext")


def test_make_editor_builds_without_loading_models():
    from splat_studio.editing import restyle

    editor = restyle.make_editor()
    assert (editor.key, editor.uses_depth, editor.pipe) == ("sdxl-depth", True, None)
    assert editor.label == models.editor_spec(None).label
    with pytest.raises(KeyError):
        restyle.make_editor("nope")
