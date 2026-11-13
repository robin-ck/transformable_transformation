import re

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
