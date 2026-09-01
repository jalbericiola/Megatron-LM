# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import hashlib
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

helpers_source = Path("megatron/core/datasets/helpers.cpp")
helpers_source_sha256 = hashlib.sha256(helpers_source.read_bytes()).hexdigest()

setup(
    ext_modules=[
        Pybind11Extension(
            "megatron.core.datasets.helpers_cpp",
            sources=[str(helpers_source)],
            language="c++",
            extra_compile_args=["-O3", "-Wall", "-std=c++17"],
            define_macros=[
                ("MEGATRON_DATASET_HELPERS_SOURCE_SHA256", f'"{helpers_source_sha256}"')
            ],
            optional=True,
        )
    ]
)
