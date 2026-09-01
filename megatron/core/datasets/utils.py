# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import importlib
import logging
import os
import platform
import re
import shlex
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Tuple

import numpy

from ..utils import log_single_rank

logger = logging.getLogger(__name__)

_HELPERS_MODULE_NAME = "megatron.core.datasets.helpers_cpp"
_HELPERS_REQUIRED_EXPORTS = (
    "build_mapping",
    "build_blocks_mapping",
    "build_sample_idx_int32",
    "build_sample_idx_int64",
    "build_blending_indices",
    "build_exhaustive_blending_indices",
)
_HELPERS_CACHE_ENV = "MEGATRON_DATASET_HELPERS_CACHE_DIR"
_HELPERS_SOURCE_DIGEST_ATTRIBUTE = "__megatron_helpers_source_sha256__"
_HELPERS_SOURCE_DIGEST_MACRO = "MEGATRON_DATASET_HELPERS_SOURCE_SHA256"
_HELPERS_BUILD_SCHEMA = 2
_HELPERS_COMPILE_TIMEOUT_SECONDS = 600
_HELPERS_DIAGNOSTIC_LIMIT = 64 * 1024
_HELPERS_THREAD_LOCK = threading.Lock()


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def _ensure_writable_directory(path: Path, *, private: bool) -> Path:
    """Create a trustworthy cache directory and verify that it is writable."""

    path = path.expanduser().absolute()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"Dataset helper cache directory must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Dataset helper cache path is not a directory: {path}")

    get_user_id = getattr(os, "getuid", None)
    if get_user_id is not None and metadata.st_uid != get_user_id():
        raise RuntimeError(f"Dataset helper cache directory is not owned by this user: {path}")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(f"Dataset helper cache directory is group- or world-writable: {path}")
    if private and stat.S_IMODE(metadata.st_mode) != 0o700:
        path.chmod(0o700)

    probe = path / f".write-test-{os.getpid()}-{uuid.uuid4().hex}"
    open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(probe, open_flags, 0o600)
    os.close(descriptor)
    probe.unlink()
    return path


def _get_helpers_cache_root() -> Path:
    """Return a secure writable cache root without writing into the source tree."""

    configured_root = os.getenv(_HELPERS_CACHE_ENV)
    if configured_root:
        try:
            return _ensure_writable_directory(Path(configured_root), private=False)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"{_HELPERS_CACHE_ENV}={configured_root!r} is not a secure writable directory"
            ) from error

    user_id = getattr(os, "getuid", lambda: "unknown")()
    cache_root = Path(tempfile.gettempdir()) / f"megatron-dataset-helpers-{user_id}"
    try:
        return _ensure_writable_directory(cache_root, private=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"No secure writable cache is available for the C++ dataset helpers at {cache_root}"
        ) from error


def _get_pybind11_identity() -> Tuple[str, str, str]:
    """Return the version and header roots that affect pybind11's generated ABI."""

    try:
        import pybind11
    except ImportError as error:
        raise RuntimeError(
            "Building the C++ dataset helpers requires pybind11 in the running Python environment"
        ) from error

    return (
        str(getattr(pybind11, "__version__", "unknown")),
        str(pybind11.get_include()),
        str(pybind11.get_include(user=True)),
    )


def _get_helpers_compiler_command() -> List[str]:
    """Return the configured C++ compiler command."""

    compiler = os.getenv("CXX") or sysconfig.get_config_var("CXX") or "c++"
    command = shlex.split(compiler)
    if not command:
        raise RuntimeError("CXX resolved to an empty compiler command")
    return command


def _get_helpers_source_digest(source: Path) -> str:
    """Return the content digest embedded in extensions built from ``source``."""

    return hashlib.sha256(source.read_bytes()).hexdigest()


def _get_helpers_build_paths() -> Tuple[Path, Path, Path]:
    """Return the source, ABI-keyed build directory, and extension path."""

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python sysconfig did not provide EXT_SUFFIX")

    source = Path(__file__).with_name("helpers.cpp")
    source_digest = _get_helpers_source_digest(source)
    pybind11_version, pybind11_include, pybind11_user_include = _get_pybind11_identity()
    build_identity = "\0".join(
        [
            str(_HELPERS_BUILD_SCHEMA),
            source_digest,
            extension_suffix,
            str(sysconfig.get_config_var("SOABI") or ""),
            str(sysconfig.get_config_var("MULTIARCH") or ""),
            str(sys.implementation.cache_tag or ""),
            sysconfig.get_platform(),
            platform.machine(),
            platform.python_compiler(),
            shlex.join(_get_helpers_compiler_command()),
            str(sysconfig.get_config_var("CC") or ""),
            str(sysconfig.get_config_var("CFLAGS") or ""),
            str(sysconfig.get_config_var("CCSHARED") or ""),
            str(sysconfig.get_config_var("LDSHARED") or ""),
            str(sysconfig.get_config_var("LDFLAGS") or ""),
            pybind11_version,
            pybind11_include,
            pybind11_user_include,
            os.getenv("CXX", ""),
            os.getenv("CPPFLAGS", ""),
            os.getenv("CXXFLAGS", ""),
            os.getenv("LDFLAGS", ""),
        ]
    )
    build_key = hashlib.sha256(build_identity.encode("utf-8")).hexdigest()
    abi_key = re.sub(r"[^A-Za-z0-9_.-]", "_", extension_suffix.lstrip("."))
    build_directory = (
        _get_helpers_cache_root() / "megatron-core" / "dataset-helpers" / abi_key / build_key
    )
    extension = build_directory / f"helpers_cpp{extension_suffix}"
    return source, build_directory, extension


def _get_helpers_compile_command(source: Path, output: Path) -> List[str]:
    """Build a compiler argv for the running Python, without ``python3-config``."""

    import pybind11

    command = _get_helpers_compiler_command()
    command.extend(shlex.split(os.getenv("CPPFLAGS", "")))
    command.extend(shlex.split(os.getenv("CXXFLAGS", "")))
    command.extend(["-O3", "-Wall", "-shared", "-std=c++17", "-fPIC"])
    source_digest = _get_helpers_source_digest(source)
    command.append(f'-D{_HELPERS_SOURCE_DIGEST_MACRO}="{source_digest}"')

    python_paths = sysconfig.get_paths()
    include_directories = [
        pybind11.get_include(),
        pybind11.get_include(user=True),
        python_paths.get("include"),
        python_paths.get("platinclude"),
    ]
    for include_directory in dict.fromkeys(include_directories):
        if include_directory:
            command.append(f"-I{include_directory}")

    if sys.platform == "darwin":
        command.extend(["-undefined", "dynamic_lookup"])

    command.extend([str(source), "-o", str(output)])
    command.extend(shlex.split(str(sysconfig.get_config_var("LDFLAGS") or "")))
    command.extend(shlex.split(os.getenv("LDFLAGS", "")))
    return command


def _build_helpers_extension(source: Path, extension: Path) -> None:
    """Compile and atomically publish the helpers extension."""

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    temporary_extension = extension.with_name(
        f".helpers_cpp-{os.getpid()}-{uuid.uuid4().hex}{extension_suffix}"
    )
    command = _get_helpers_compile_command(source, temporary_extension)
    try:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_HELPERS_COMPILE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Timed out after "
                f"{_HELPERS_COMPILE_TIMEOUT_SECONDS} seconds compiling the C++ dataset helpers"
            ) from error
        if result.returncode != 0:
            output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
            if len(output) > _HELPERS_DIAGNOSTIC_LIMIT:
                output = "[compiler output truncated]\n" + output[-_HELPERS_DIAGNOSTIC_LIMIT:]
            raise RuntimeError(
                "Failed to compile the C++ dataset helpers with "
                f"{shlex.join(command)}" + (f"\n{output}" if output else "")
            )
        if not temporary_extension.is_file() or temporary_extension.stat().st_size == 0:
            raise RuntimeError("The C++ dataset helper compiler did not produce an extension")

        temporary_extension.chmod(0o755)
        os.replace(temporary_extension, extension)
        try:
            directory_descriptor = os.open(extension.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # Some network filesystems do not support fsync on directories. The atomic replace
            # above still prevents readers from observing a partially written extension.
            pass
    finally:
        temporary_extension.unlink(missing_ok=True)


@contextmanager
def _helpers_build_lock(lock_path: Path):
    """Serialize publishers on both local and shared filesystems."""

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - Megatron's supported platforms provide it.
        raise RuntimeError("Dataset helper compilation requires POSIX file locking") from error

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_helpers_extension(build_directory: Path) -> ModuleType:
    """Import the cached extension under its canonical package name."""

    loaded_module = sys.modules.get(_HELPERS_MODULE_NAME)
    if loaded_module is not None:
        return loaded_module

    datasets_package = importlib.import_module("megatron.core.datasets")
    build_directory_string = str(build_directory)
    if build_directory_string in datasets_package.__path__:
        datasets_package.__path__.remove(build_directory_string)
    datasets_package.__path__.insert(0, build_directory_string)
    importlib.invalidate_caches()
    return importlib.import_module(_HELPERS_MODULE_NAME)


def _validate_helpers_extension(
    module: ModuleType, *, expected_source_digest: str | None = None
) -> ModuleType:
    """Require the expected source identity and complete API before accepting an extension."""

    missing_exports = [
        export
        for export in _HELPERS_REQUIRED_EXPORTS
        if not callable(getattr(module, export, None))
    ]
    if missing_exports:
        raise ImportError(
            f"{_HELPERS_MODULE_NAME} is missing required exports: {', '.join(missing_exports)}"
        )
    if expected_source_digest is not None:
        extension_source_digest = getattr(module, _HELPERS_SOURCE_DIGEST_ATTRIBUTE, None)
        if extension_source_digest != expected_source_digest:
            reported_digest = (
                repr(extension_source_digest)
                if extension_source_digest is not None
                else "not reported by this extension"
            )
            raise ImportError(
                f"{_HELPERS_MODULE_NAME} was not built from the current helpers.cpp "
                f"(expected SHA-256 {expected_source_digest}, got {reported_digest})"
            )
    return module


def _discard_helpers_extension(module: ModuleType | None = None) -> None:
    """Remove a failed canonical import before trying a cache-built extension."""

    loaded_module = sys.modules.get(_HELPERS_MODULE_NAME)
    if module is None or loaded_module is module:
        sys.modules.pop(_HELPERS_MODULE_NAME, None)

    datasets_package = sys.modules.get("megatron.core.datasets")
    package_module = (
        getattr(datasets_package, "helpers_cpp", None) if datasets_package is not None else None
    )
    if package_module is not None and (module is None or package_module is module):
        delattr(datasets_package, "helpers_cpp")


def compile_helpers() -> ModuleType:
    """Build and import the C++ dataset helpers from a writable, ABI-keyed cache.

    Every rank may call this function. A process lock and atomic publication make concurrent calls
    safe whether the cache is node-local or shared by multiple nodes. Each caller imports the
    extension itself so ``megatron.core.datasets.helpers_cpp`` is available in every process.

    Returns:
        The imported ``megatron.core.datasets.helpers_cpp`` extension module.
    """

    with _HELPERS_THREAD_LOCK:
        package_source = Path(__file__).with_name("helpers.cpp")
        try:
            package_source_digest = _get_helpers_source_digest(package_source)
        except FileNotFoundError:
            # Binary-only installations can rely on their packaged extension. Editable installs
            # include helpers.cpp, so they take the content-identity check below.
            package_source_digest = None

        loaded_module = sys.modules.get(_HELPERS_MODULE_NAME)
        if loaded_module is not None:
            try:
                return _validate_helpers_extension(
                    loaded_module, expected_source_digest=package_source_digest
                )
            except ImportError:
                _discard_helpers_extension(loaded_module)

        try:
            return _validate_helpers_extension(
                importlib.import_module(_HELPERS_MODULE_NAME),
                expected_source_digest=package_source_digest,
            )
        except ImportError:
            _discard_helpers_extension()

        source, build_directory, extension = _get_helpers_build_paths()
        source_digest = _get_helpers_source_digest(source)
        build_directory.mkdir(parents=True, exist_ok=True)

        if extension.is_file():
            try:
                return _validate_helpers_extension(
                    _load_helpers_extension(build_directory), expected_source_digest=source_digest
                )
            except ImportError:
                _discard_helpers_extension()
                # Recheck under the publisher lock before replacing a stale or damaged cache entry.
                pass

        with _helpers_build_lock(build_directory / ".build.lock"):
            if extension.is_file():
                try:
                    return _validate_helpers_extension(
                        _load_helpers_extension(build_directory),
                        expected_source_digest=source_digest,
                    )
                except ImportError:
                    _discard_helpers_extension()
                    extension.unlink(missing_ok=True)

            _build_helpers_extension(source, extension)
            try:
                return _validate_helpers_extension(
                    _load_helpers_extension(build_directory),
                    expected_source_digest=_get_helpers_source_digest(source),
                )
            except ImportError as error:
                _discard_helpers_extension()
                extension.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Built dataset helpers at {extension}, but Python could not import them"
                ) from error


def compile_helpers_distributed() -> ModuleType:
    """Build and import dataset helpers on every rank with convergent failure handling.

    A node-local cache requires one caller on every node, and importing a native extension is
    process-local, so restricting this operation to local rank zero is insufficient. When a
    process group exists, every rank attempts the operation and then participates in a one-element
    tensor all-reduce. This makes all ranks raise if any peer fails without using an object
    collective during bootstrap.

    Returns:
        The imported ``megatron.core.datasets.helpers_cpp`` extension module.
    """

    import torch

    if not torch.distributed.is_initialized():
        return compile_helpers()

    rank = torch.distributed.get_rank()
    module = None
    failure = None
    try:
        module = compile_helpers()
    except Exception as error:  # Keep peers out of a one-sided post-build barrier.
        failure = f"rank {rank}: {type(error).__name__}: {error}"

    backend = torch.distributed.get_backend()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if str(backend).lower() == str(torch.distributed.Backend.NCCL).lower()
        else torch.device("cpu")
    )
    failure_flag = torch.tensor(int(failure is not None), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(failure_flag, op=torch.distributed.ReduceOp.MAX)
    if failure_flag.item():
        if failure is not None:
            raise RuntimeError(f"Failed to compile the C++ dataset helpers: {failure}")
        raise RuntimeError("Failed to compile the C++ dataset helpers on another rank")

    assert module is not None
    return module


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """

    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    # pylint: disable=line-too-long
    """Get the blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend
    from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either
            (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or
            (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    # pylint: enable=line-too-long
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset
