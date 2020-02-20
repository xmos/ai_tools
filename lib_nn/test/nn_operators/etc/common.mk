


ifeq ($(OS),Windows_NT)
mkdir_cmd = @test -d $(subst /,\,$(dir $(1))) || mkdir $(subst /,\,$(dir $(1)))
else
mkdir_cmd = @mkdir -p $(dir $(1))
endif

check_defined = $(strip $(foreach 1,$1, $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = $(if $(value $1),, $(error Undefined $1$(if $2, ($2))))

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))