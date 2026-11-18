from splat_studio.space import tool_options
from splat_studio.space.jobs import LocalStore


def test_clean_keeps_known_ids_in_tab_order():
    assert tool_options.clean(["futures", "nope", "history", "futures"]) == ["history", "futures"]
    assert tool_options.clean("history") == []
    assert tool_options.clean(None) == []


def test_parse_tolerates_bad_text():
    assert tool_options.parse('["preview"]') == ["preview"]
    assert tool_options.parse("not json") == []
    assert tool_options.parse(None) == []
    assert tool_options.parse('{"a": 1}') == []


def test_load_default_is_all_visible(tmp_path):
    assert tool_options.load(LocalStore(tmp_path)) == []


def test_save_and_load_round_trip_keeps_other_keys(tmp_path):
    store = LocalStore(tmp_path)
    store.write_json(tool_options.PATH, {"other": 1})
    assert tool_options.save(store, ["futures", "bogus", "edit"]) == ["edit", "futures"]
    assert tool_options.load(store) == ["edit", "futures"]
    assert store.read_json(tool_options.PATH)["other"] == 1


def test_load_ignores_broken_file(tmp_path):
    store = LocalStore(tmp_path)
    store.write_json(tool_options.PATH, {"hidden_tools": "futures"})
    assert tool_options.load(store) == []


def test_every_tool_has_its_sub_tab():
    """The page script finds tools by these elem_ids and labels (space/app.py)."""
    from pathlib import Path

    app = (Path(__file__).resolve().parent.parent / "space" / "app.py").read_text()
    for tool, label in tool_options.TOOLS.items():
        assert f'gr.Tab("{label}", elem_id="tool-{tool}")' in app
    assert 'elem_id="scene-tools"' in app and 'elem_id="scene-tool-tabs"' in app
