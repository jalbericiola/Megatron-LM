# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import subprocess
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from megatron.core.datasets import utils
from megatron.core.utils import log_single_rank as core_log_single_rank


def make_helpers_module(source_digest=None):
    module = ModuleType(utils._HELPERS_MODULE_NAME)
    for export in utils._HELPERS_REQUIRED_EXPORTS:
        setattr(module, export, lambda: None)
    if source_digest is None:
        source = Path(utils.__file__).with_name("helpers.cpp")
        source_digest = utils._get_helpers_source_digest(source)
    setattr(module, utils._HELPERS_SOURCE_DIGEST_ATTRIBUTE, source_digest)
    return module


def test_preserves_public_log_single_rank_reexport():
    """Dataset configuration imports this legacy symbol from datasets.utils."""

    assert utils.log_single_rank is core_log_single_rank


@pytest.fixture(autouse=True)
def clean_helpers_module(monkeypatch):
    """Keep the process-global extension import state isolated between tests."""

    datasets_package = sys.modules["megatron.core.datasets"]
    real_importlib = utils.importlib

    def import_without_packaged_helpers(module_name, package=None):
        if module_name == utils._HELPERS_MODULE_NAME:
            raise ImportError("packaged dataset helpers are isolated by this test fixture")
        return real_importlib.import_module(module_name, package)

    # Other dataset tests may have compiled the extension already. That permanently prepends its
    # cache directory to the package path, so deleting only sys.modules does not isolate these
    # tests: the next canonical import simply finds the real extension again. Give utils a local
    # importlib facade that treats the packaged extension as absent unless a test explicitly
    # supplies one, and make package-path mutations local to each test as well.
    monkeypatch.setattr(
        utils,
        "importlib",
        SimpleNamespace(
            import_module=import_without_packaged_helpers,
            invalidate_caches=real_importlib.invalidate_caches,
        ),
    )
    monkeypatch.setattr(datasets_package, "__path__", list(datasets_package.__path__))
    monkeypatch.delitem(sys.modules, utils._HELPERS_MODULE_NAME, raising=False)
    monkeypatch.delattr(datasets_package, "helpers_cpp", raising=False)


def test_build_path_uses_writable_abi_keyed_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(utils._HELPERS_CACHE_ENV, str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "pybind11",
        SimpleNamespace(
            __version__="2.13.6",
            get_include=lambda user=False: "/pybind/user" if user else "/pybind/system",
        ),
    )

    source, build_directory, extension = utils._get_helpers_build_paths()

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    abi_key = extension_suffix.lstrip(".")
    assert source == Path(utils.__file__).with_name("helpers.cpp")
    assert build_directory.is_relative_to(tmp_path)
    assert abi_key in build_directory.parts
    assert extension == build_directory / f"helpers_cpp{extension_suffix}"
    assert source.parent not in build_directory.parents


def test_build_key_changes_with_pybind_and_compiler_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(utils._HELPERS_CACHE_ENV, str(tmp_path))
    pybind11 = SimpleNamespace(
        __version__="2.13.6",
        get_include=lambda user=False: "/pybind/user" if user else "/pybind/system",
    )
    monkeypatch.setitem(sys.modules, "pybind11", pybind11)
    monkeypatch.setenv("CXX", "c++-one")
    first_directory = utils._get_helpers_build_paths()[1]

    pybind11.__version__ = "2.13.7"
    second_directory = utils._get_helpers_build_paths()[1]
    monkeypatch.setenv("CXX", "c++-two")
    third_directory = utils._get_helpers_build_paths()[1]

    assert len({first_directory, second_directory, third_directory}) == 3


def test_default_cache_is_uid_private_and_node_local(tmp_path, monkeypatch):
    monkeypatch.delenv(utils._HELPERS_CACHE_ENV, raising=False)
    monkeypatch.setattr(utils.tempfile, "gettempdir", lambda: os.fspath(tmp_path))

    cache_root = utils._get_helpers_cache_root()

    assert cache_root.parent == tmp_path
    assert cache_root.stat().st_mode & 0o777 == 0o700
    assert not cache_root.is_symlink()


def test_configured_cache_rejects_symlink(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    cache_link = tmp_path / "cache-link"
    cache_link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv(utils._HELPERS_CACHE_ENV, os.fspath(cache_link))

    with pytest.raises(RuntimeError, match="secure writable directory"):
        utils._get_helpers_cache_root()


def test_configured_cache_rejects_group_or_world_writable_directory(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_root.chmod(0o777)
    monkeypatch.setenv(utils._HELPERS_CACHE_ENV, os.fspath(cache_root))

    with pytest.raises(RuntimeError, match="secure writable directory"):
        utils._get_helpers_cache_root()


def test_compile_command_uses_running_python_without_python3_config(tmp_path, monkeypatch):
    pybind11 = SimpleNamespace(
        __version__="2.13.6",
        get_include=lambda user=False: "/pybind/user" if user else "/pybind/system",
    )
    monkeypatch.setitem(sys.modules, "pybind11", pybind11)
    monkeypatch.setenv("CXX", "cache-wrapper c++")

    source = tmp_path / "helpers.cpp"
    source.write_text("// current source")
    command = utils._get_helpers_compile_command(
        source, tmp_path / f"helpers_cpp{sysconfig.get_config_var('EXT_SUFFIX')}"
    )

    assert command[:2] == ["cache-wrapper", "c++"]
    assert "python3-config" not in " ".join(command)
    assert "-I/pybind/system" in command
    assert "-I/pybind/user" in command
    assert f"-I{sysconfig.get_paths()['include']}" in command
    assert command[command.index("-o") + 1].endswith(sysconfig.get_config_var("EXT_SUFFIX"))
    source_digest = utils._get_helpers_source_digest(source)
    assert f'-D{utils._HELPERS_SOURCE_DIGEST_MACRO}="{source_digest}"' in command


def test_build_atomically_publishes_compiler_output(tmp_path, monkeypatch):
    extension = tmp_path / f"helpers_cpp{sysconfig.get_config_var('EXT_SUFFIX')}"

    def fake_run(command, **kwargs):
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "check": False,
            "timeout": utils._HELPERS_COMPILE_TIMEOUT_SECONDS,
        }
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"extension")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        utils,
        "_get_helpers_compile_command",
        lambda source, output: ["c++", str(source), "-o", str(output)],
    )
    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    utils._build_helpers_extension(tmp_path / "helpers.cpp", extension)

    assert extension.read_bytes() == b"extension"
    assert not list(tmp_path.glob(".helpers_cpp-*"))


def test_build_timeout_is_bounded_and_cleans_temporary_output(tmp_path, monkeypatch):
    extension = tmp_path / f"helpers_cpp{sysconfig.get_config_var('EXT_SUFFIX')}"

    monkeypatch.setattr(
        utils,
        "_get_helpers_compile_command",
        lambda source, output: ["c++", str(source), "-o", str(output)],
    )

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(utils.subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="Timed out after 600 seconds"):
        utils._build_helpers_extension(tmp_path / "helpers.cpp", extension)

    assert not extension.exists()
    assert not list(tmp_path.glob(".helpers_cpp-*"))


def test_build_lock_serializes_concurrent_publishers(tmp_path):
    lock_path = tmp_path / ".build.lock"
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def hold_first_lock():
        with utils._helpers_build_lock(lock_path):
            first_acquired.set()
            assert release_first.wait(timeout=5)

    def take_second_lock():
        assert first_acquired.wait(timeout=5)
        with utils._helpers_build_lock(lock_path):
            second_acquired.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(hold_first_lock)
        second = executor.submit(take_second_lock)
        assert first_acquired.wait(timeout=5)
        assert not second_acquired.wait(timeout=0.1)
        release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert second_acquired.is_set()


def test_compile_helpers_prefers_complete_packaged_extension(monkeypatch):
    module = make_helpers_module()

    monkeypatch.setattr(
        utils.importlib,
        "import_module",
        lambda module_name: (
            module
            if module_name == utils._HELPERS_MODULE_NAME
            else pytest.fail(f"unexpected import: {module_name}")
        ),
    )
    monkeypatch.setattr(
        utils,
        "_get_helpers_build_paths",
        lambda: pytest.fail("a packaged extension must not require a compiler or pybind11"),
    )

    assert utils.compile_helpers() is module


def test_compile_helpers_rebuilds_extension_from_edited_source(tmp_path, monkeypatch):
    source = tmp_path / "helpers.cpp"
    source.write_text("// edited source")
    source_digest = utils._get_helpers_source_digest(source)
    stale_module = make_helpers_module("0" * 64)
    rebuilt_module = make_helpers_module(source_digest)
    build_directory = tmp_path / "build"
    extension = build_directory / f"helpers_cpp{sysconfig.get_config_var('EXT_SUFFIX')}"
    build_count = 0

    monkeypatch.setattr(utils, "__file__", str(tmp_path / "utils.py"))
    monkeypatch.setattr(
        utils, "_get_helpers_build_paths", lambda: (source, build_directory, extension)
    )
    monkeypatch.setattr(utils.importlib, "import_module", lambda _: stale_module)

    def fake_build(build_source, build_extension):
        nonlocal build_count
        assert build_source == source
        build_count += 1
        build_extension.write_bytes(b"rebuilt extension")

    def fake_load(directory):
        assert directory == build_directory
        monkeypatch.setitem(sys.modules, utils._HELPERS_MODULE_NAME, rebuilt_module)
        return rebuilt_module

    monkeypatch.setattr(utils, "_build_helpers_extension", fake_build)
    monkeypatch.setattr(utils, "_load_helpers_extension", fake_load)

    assert utils.compile_helpers() is rebuilt_module
    assert build_count == 1


def test_compile_helpers_accepts_extension_built_from_current_source(tmp_path, monkeypatch):
    source = tmp_path / "helpers.cpp"
    source.write_text("// current source")
    module = make_helpers_module(utils._get_helpers_source_digest(source))

    monkeypatch.setattr(utils, "__file__", str(tmp_path / "utils.py"))
    monkeypatch.setattr(utils.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(
        utils,
        "_get_helpers_build_paths",
        lambda: pytest.fail("a source-matched extension must not be rebuilt"),
    )

    assert utils.compile_helpers() is module


def test_validate_helpers_extension_requires_all_six_exports():
    module = make_helpers_module()
    delattr(module, "build_blocks_mapping")

    with pytest.raises(ImportError, match="build_blocks_mapping"):
        utils._validate_helpers_extension(module)


def test_validate_helpers_extension_rejects_legacy_binary_without_source_digest():
    module = make_helpers_module()
    delattr(module, utils._HELPERS_SOURCE_DIGEST_ATTRIBUTE)
    source = Path(utils.__file__).with_name("helpers.cpp")

    with pytest.raises(ImportError, match="not reported by this extension"):
        utils._validate_helpers_extension(
            module, expected_source_digest=utils._get_helpers_source_digest(source)
        )


def test_compile_helpers_builds_only_in_cache_and_registers_module(tmp_path, monkeypatch):
    source = Path(utils.__file__).with_name("helpers.cpp")
    build_directory = tmp_path / "build"
    extension = build_directory / f"helpers_cpp{sysconfig.get_config_var('EXT_SUFFIX')}"
    module = make_helpers_module()
    built_paths = []

    monkeypatch.setattr(
        utils, "_get_helpers_build_paths", lambda: (source, build_directory, extension)
    )

    def fake_build(build_source, build_extension):
        built_paths.append((build_source, build_extension))
        build_extension.write_bytes(b"extension")

    def fake_load(directory):
        assert directory == build_directory
        monkeypatch.setitem(sys.modules, utils._HELPERS_MODULE_NAME, module)
        return module

    monkeypatch.setattr(utils, "_build_helpers_extension", fake_build)
    monkeypatch.setattr(utils, "_load_helpers_extension", fake_load)

    assert utils.compile_helpers() is module
    assert built_paths == [(source, extension)]
    assert extension.is_relative_to(tmp_path)


def test_concurrent_compile_helpers_has_one_publisher(tmp_path, monkeypatch):
    source = tmp_path / "helpers.cpp"
    source.write_text("// source")
    build_directory = tmp_path / "build"
    extension = build_directory / f"helpers_cpp{sysconfig.get_config_var('EXT_SUFFIX')}"
    module = make_helpers_module(utils._get_helpers_source_digest(source))
    build_count = 0

    monkeypatch.setattr(
        utils, "_get_helpers_build_paths", lambda: (source, build_directory, extension)
    )

    def fake_build(build_source, build_extension):
        nonlocal build_count
        assert build_source == source
        build_count += 1
        build_extension.write_bytes(b"extension")

    def fake_load(directory):
        assert directory == build_directory
        monkeypatch.setitem(sys.modules, utils._HELPERS_MODULE_NAME, module)
        return module

    monkeypatch.setattr(utils, "_build_helpers_extension", fake_build)
    monkeypatch.setattr(utils, "_load_helpers_extension", fake_load)

    with ThreadPoolExecutor(max_workers=8) as executor:
        modules = list(executor.map(lambda _: utils.compile_helpers(), range(32)))

    assert modules == [module] * 32
    assert build_count == 1


def test_load_helpers_extension_prepends_cache_to_package_path(tmp_path, monkeypatch):
    package = SimpleNamespace(__path__=["/read-only/source/megatron/core/datasets"])
    module = make_helpers_module()

    def fake_import(module_name):
        if module_name == "megatron.core.datasets":
            return package
        assert module_name == utils._HELPERS_MODULE_NAME
        assert package.__path__[0] == os.fspath(tmp_path)
        return module

    monkeypatch.setattr(utils.importlib, "import_module", fake_import)

    assert utils._load_helpers_extension(tmp_path) is module
    assert package.__path__ == [os.fspath(tmp_path), "/read-only/source/megatron/core/datasets"]
