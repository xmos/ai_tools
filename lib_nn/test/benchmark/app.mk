

######
### [required]
### Application name. Used as output file name.
###
APP_NAME = benchmark

######
### [required if $(PLATFORM) is xcore]
### xcore target device
###
TARGET_DEVICE = XU316-1024-FB265-C32

######
### [optional] List of libraries on which this application depends
###
DEPENDENCIES := lib_nn

######
### [currently required]
### Paths to the dependencies, because we don't have a way of searching
### for them at the moment.
###
lib_nn_PATH := ../../lib_nn

###### 
### [optional] Directories, relative to the application project, to look for source files.
###            defaults to nothing.
SOURCE_DIRS := src

###### 
### [optional] Source file extentions. Defaults to: c cc xc cpp S
###
SOURCE_FILE_EXTS := c cc cpp

ifeq ($(PLATFORM),xcore)
  SOURCE_FILE_EXTS += xc
endif

######
### [optional] List of source files to compile.
###            
# SOURCE_FILES :=  $(wildcard src/*.c src/*.xc)

######
### [optional] list of static libraries that
### should be linked into the executable
###
# LIBRARIES := foo.a

# If the application makefile sets this to any value other than 1, no static 
# libraries (.a files) will be created for dependencies, and the executable
# will be linked directly against library object files.
#
BUILD_STATIC_LIBRARIES := 0

LD_FLAGS += -fcmdline-buffer-bytes=4000

ifeq ($(DEBUG),1)
  $(info Debug enabled..)
  CC_FLAGS += -O0
endif


######
### [required]
### Default make target. Use this for a project description?
###
app_help:
	$(info ****************************************************************)
	$(info benchmark: Performance tests for the functions in lib_nn        )
	$(info                                                                 )
	$(info make targets:                                                   )
	$(info |     help:    Display this message (default)                   )
	$(info |      all:    Build application                                )
	$(info |    clean:    Remove build files and folders from project      )
	$(info |      run:    Run and trace the tests. Process trace.          )
	$(info ****************************************************************)

DUMP_DIR := dump
APP_XE_FILE := $(EXE_DIR)/$(APP_NAME).xe
PERF_FILE := $(DUMP_DIR)/perf.txt

# TRACE_LOG := $(DUMP_DIR)/trace.$(CONFIG).log

ifndef FUNC
  FUNC_LIST := vpu_memcpy requantize_16_to_8 lookup8 conv2d_deep nn_conv2d_hstrip_deep avgpool2d bnn_conv2d_bin_output
else
  FUNC_LIST := $(FUNC)
endif

run_traces: executable
	$(call mkdir_cmd, $(PERF_FILE))
	python python/run_traces.py $(TRACE_FLAGS) --out-dir $(DUMP_DIR) $(APP_XE_FILE) $(FUNC_LIST)

run: run_traces;

clean: app_clean

app_clean:
	rm -rf trace.local.tmp