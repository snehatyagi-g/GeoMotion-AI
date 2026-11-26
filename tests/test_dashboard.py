"""Basic functional checks for the Streamlit dashboard and its map helpers."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import parse_linestring_wkt


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_overview_renders_without_app_exceptions():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("Synthetic benchmark" in warning.value for warning in app.warning)
    assert any("GeoMotion AI" in title.value for title in app.title)


def test_all_six_dashboard_sections_are_registered():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    # Page navigation is declared in app.py; inspect the public page declarations
    # instead of depending on Streamlit's private navigation tree representation.
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    expected = {"Overview", "Dataset Explorer", "Road Network", "Map Matching", "Algorithm Comparison", "Results and Export"}
    assert all(f'title="{name}"' in source for name in expected)


def test_linestring_parser_returns_pydeck_longitude_latitude_positions():
    assert parse_linestring_wkt("LINESTRING (77.1 28.2, 77.3 28.4)") == [[77.1, 28.2], [77.3, 28.4]]
    assert parse_linestring_wkt("invalid") == []


@pytest.mark.parametrize("page_function", [
    "render_dataset_explorer", "render_road_network", "render_map_matching",
    "render_algorithm_comparison", "render_results_export",
])
def test_dashboard_sections_render_without_exceptions(page_function):
    app = AppTest.from_string(f"import app\napp.{page_function}()", default_timeout=45).run()
    assert not app.exception


def test_map_matching_can_run_an_existing_python_matcher():
    app = AppTest.from_string("import app\napp.render_map_matching()", default_timeout=60).run()
    app.selectbox(key="matching_interval").select(120).run()
    app.button(key="run_matchers").click().run()
    assert not app.exception
    assert "dashboard_match_result" in app.session_state
