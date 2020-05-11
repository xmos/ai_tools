

Unity_PATH ?= ./
LIB_PATH := $(abspath $(Unity_PATH))

INCLUDES += $(LIB_PATH)/src


###### 
### [optional] Directories, relative to the dependency folder, to look for source files.
###            defaults to nothing.
SOURCE_DIRS := src

###### 
### [optional] Source file extentions. Defaults to: c cc xc cpp S
###
SOURCE_FILE_EXTS := c

######
### [optional] List of source files to compile.
###            
# SOURCE_FILES :=  $(wildcard src/*.c src/*.xc)

######
### [optional] list of static libraries that
### should be linked into the executable
###
# LIBRARIES := foo.a
