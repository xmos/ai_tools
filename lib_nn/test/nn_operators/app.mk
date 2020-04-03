

######
### [required]
### Application name. Used as output file name.
###
APP_NAME = nn_operators_test

######
### [required if $(PLATFORM) is xcore]
### xcore target device
###
TARGET_DEVICE = XU316-1024-QF60-C20
#TARGET_DEVICE = XU316-1024-FB265-C32

######
### [optional] List of libraries on which this application depends
###
DEPENDENCIES := lib_nn Unity

######
### [currently required]
### Paths to the dependencies, because we don't have a way of searching
### for them at the moment.
###
lib_nn_PATH := ../../lib_nn
Unity_PATH := ../../Unity

###### 
### [optional] Directories, relative to the application project, to look for source files.
###            defaults to nothing.
SOURCE_DIRS := src

###### 
### [optional] Source file extentions. Defaults to: c cc xc cpp S
###
# SOURCE_FILE_EXTS := c cc xc cpp S

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

######
### [required]
### Default make target. Use this for a project description?
###
app_help:
	$(info *********************************************************)
	$(info nn_operators_test: Unit tests for the functions in lib_nn)
	$(info                                                          )
	$(info make targets:                                            )
	$(info |   help:    Display this message (default)              )
	$(info |    all:    Build application                           )
	$(info |  clean:    Remove build files and folders from project )
	$(info |    run:    Run the unit tests in xsim                  )
	$(info *********************************************************)


#####################################
### Application-specific targets
#####################################

run: build
	xsim $(APP_EXE_FILE)