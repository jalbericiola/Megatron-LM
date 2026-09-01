PYTHON ?= python3

CXXFLAGS += -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell $(PYTHON) -m pybind11 --includes)

LIBNAME = helpers_cpp
LIBEXT = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp
SOURCE_SHA256 = $(shell $(PYTHON) -c 'import hashlib; print(hashlib.sha256(open("$(SRC)", "rb").read()).hexdigest())')
CPPFLAGS += -DMEGATRON_DATASET_HELPERS_SOURCE_SHA256=\"$(SOURCE_SHA256)\"

default: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
