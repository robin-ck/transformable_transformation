"""Editor instructions (edit-editor-instructions): visible, changeable text added to prompts."""

import pytest

from splat_studio.editing import area, instructions, masking
from splat_studio.space import futures, viewpoints


def test_defaults_give_the_same_prompts_as_before():
    assert masking.object_prompt("make it green", "chair").endswith(
        "Keep the shape, material, texture, shading and lighting of the chair. "
        "Keep everything else unchanged."
    )
    assert "Fill the space in the middle" in area.area_prompt("remove the fence")


def test_own_texts_replace_the_defaults_and_empty_adds_nothing():
    texts = {"object_change": "Only recolour {target}.", "area_change": ""}
    assert masking.object_prompt("make it green", "chair", texts) == (
        "make the chair green. Only recolour the chair."
    )
    assert area.area_prompt("a willow", texts) == "a willow."
    assert "Fill the space" in area.area_prompt("remove the fence", texts)  # default kept


def test_clean_keeps_only_changes_and_refuses_bad_texts():
    d = instructions.DEFAULTS
    assert instructions.clean(dict(d)) == {}
    changed = instructions.clean({**d, "negative": "  cartoon,   text "})
    assert changed == {"negative": "cartoon, text"}
    with pytest.raises(instructions.InstructionError, match="target"):
        instructions.clean({"object_change": "Keep {thing}."})
    with pytest.raises(instructions.InstructionError, match="limit"):
        instructions.clean({"negative": "x" * (instructions.MAX_TEXT + 1)})
    with pytest.raises(instructions.InstructionError, match="Unknown"):
        instructions.clean({"nope": "x"})


def test_setup_stores_instructions_and_futures_record_the_full_set():
    setup = viewpoints.default_setup()
    setup.instructions = {"negative": "cartoon"}
    again = viewpoints.Setup.from_dict(setup.to_dict())
    assert again.instructions == {"negative": "cartoon"}
    assert viewpoints.Setup.from_dict({}).instructions == {}
    vp = viewpoints.Viewpoint("v1", "bridge", 1, {"c2w": [[1, 0, 0, 0]] * 4}, 0, 0.0)
    params = futures.future_params(vp, "viewpoints/s/v1", None, "reeds", texts=setup.instructions)
    assert params["instructions"]["negative"] == "cartoon"
    assert params["instructions"]["area_change"] == instructions.DEFAULTS["area_change"]


def test_sent_preview_shows_what_the_editor_gets():
    md = futures.sent_preview(None, "reeds", "sdxl-depth", texts={"negative": "cartoon"})
    assert "reeds" in md and "**Negative prompt:** cartoon" in md
    md = futures.sent_preview(None, "make it red", "flux-kontext", "bench")
    assert "make the bench red. Keep the shape" in md and "Negative" not in md
    md = futures.sent_preview(None, "a willow", "flux-kontext", has_area=True)
    assert "Make the change in the middle" in md
    assert "Type what should change" in futures.sent_preview(None, "", "sdxl-depth")


def test_removal_words_and_pronouns_can_be_changed_or_turned_off():
    assert masking.is_removal("take   away the fence")  # default list, any spacing
    texts = {"removal_words": "Remove, weg", "pronouns": "it, this one"}
    assert masking.is_removal("bench weg", texts) and not masking.is_removal("delete it", texts)
    assert not masking.is_removal("remove the bench", {"removal_words": ""})
    assert masking.object_prompt("paint this one red", "bench", texts).startswith(
        "paint the bench red."
    )
    p = masking.object_prompt("make it red", "bench", {"pronouns": ""})
    assert p.startswith("make it red (the bench).")
    assert "Fill the space in the middle" in area.area_prompt("bench weg", texts)
    assert instructions.clean({"pronouns": " It ,, THEM , this  one"}) == {
        "pronouns": "it, them, this one"
    }


def test_preview_negative_is_kept_apart_from_the_futures_one():
    assert "preview_negative" not in instructions.FUTURE_KEYS
    texts = instructions.resolve({"preview_negative": "cartoon"})
    assert texts["negative"] == instructions.DEFAULTS["negative"]
    assert texts["preview_negative"] == "cartoon"
