"""Exercise the real path resolvers without importing models or experiment jobs."""
import ast
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
RETIRED = "C:/Users/Jiabao Zhao/Documents/Codex/2026-06-02/can-you-add-this-skill-to"
SUFFIX = "usb-cable-workspace/usb_connector_with_cable.stl"
RESOLVERS = (
    ("experiments/cad_grasp_geometry_20261026/inputs.py", "local"),
    ("experiments/cad_grasp_geometry_20261026/gallery.py", "local_path"),
    ("experiments/nist12_qwen_clip_reranking_20261023/run_dino.py", "local"),
    ("experiments/nist12_qwen_clip_reranking_20261023/analyze.py", "local"),
    ("experiments/nist12_qwen_clip_reranking_20261023/depth_verification_20261023/run_trial.py", "local"),
    ("experiments/nist12_qwen_clip_reranking_20261023/depth_verification_20261023/analyze_results.py", "local"),
    ("experiments/nist12_qwen_clip_reranking_20261023/depth_verification_20261023/build_depth_figures.py", "local"),
    ("experiments/nist12_fullscene_benchmarks_20261023/common.py", "path"),
    ("experiments/nist12_matching_benchmarks_20261023/run_benchmarks.py", "local"),
    ("experiments/nist12_matching_benchmarks_20261023/analyze_benchmarks.py", "local"),
    ("experiments/nist12_matching_benchmarks_20261023/independent_verification.py", "local"),
    ("experiments/nist12_robustness_20261024/common.py", "path"),
    ("experiments/nist12_robustness_20261024/baselines/common.py", "path"),
    ("experiments/nist12_robustness_20261024/capture/render_scenes.py", "path"),
)


@pytest.fixture(params=("win32", "linux"))
def platform_paths(request):
    if request.param == "win32":
        return request.param, PureWindowsPath, PureWindowsPath("D:/repo")
    return request.param, PurePosixPath, PurePosixPath("/mnt/d/repo")


@pytest.fixture(params=RESOLVERS, ids=lambda item: item[0])
def resolver(request, platform_paths):
    filename, name = request.param
    platform, path_type, root = platform_paths
    tree = ast.parse((REPO / filename).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"Path": path_type, "REPO": root, "ROOT": root,
                 "sys": SimpleNamespace(platform=platform),
                 "os": SimpleNamespace(name="nt" if platform == "win32" else "posix")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name], platform, path_type, root


def test_windows_backslash_and_wsl_paths_reach_the_same_preserved_mesh(resolver):
    local, _, _, root = resolver
    expected = root / "outputs/datasets/nist_task_board/legacy_workspace_inputs" / SUFFIX
    for value in (RETIRED + "/" + SUFFIX, (RETIRED + "/" + SUFFIX).replace("/", "\\"),
                  "/mnt/c" + RETIRED[2:] + "/" + SUFFIX):
        assert local(value) == expected, value


def test_unrelated_c_paths_and_workspace_name_siblings_keep_existing_behavior(resolver):
    local, platform, path_type, _ = resolver
    for value in ("C:/Users/Another User/part.stl", RETIRED + "-backup/part.stl"):
        expected = value if platform == "win32" else "/mnt/c" + value[2:]
        assert local(value) == path_type(expected), value


def test_relative_and_native_absolute_paths_keep_existing_behavior(resolver):
    local, platform, path_type, root = resolver
    assert local("CAD/part.stl") == root / "CAD/part.stl"
    absolute = "D:/assets/part.stl" if platform == "win32" else "/mnt/d/assets/part.stl"
    assert local(absolute) == path_type(absolute)


def test_migrated_suffix_cannot_escape_the_preserved_input_directory(resolver):
    local, _, _, _ = resolver
    for suffix in ("../outside.stl", "usb-cable-workspace/../../outside.stl", "D:/outside.stl"):
        for prefix in (RETIRED, "/mnt/c" + RETIRED[2:]):
            with pytest.raises(ValueError, match="Unsafe retired-workspace path suffix"):
                local(prefix + "/" + suffix)
        with pytest.raises(ValueError, match="Unsafe retired-workspace path suffix"):
            local((RETIRED + "/" + suffix).replace("/", "\\"))


def test_exact_retired_workspace_root_maps_to_preserved_root(resolver):
    local, _, _, root = resolver
    expected = root / "outputs/datasets/nist_task_board/legacy_workspace_inputs"
    assert local(RETIRED) == expected
    assert local("/mnt/c" + RETIRED[2:]) == expected
