PROJECT := nyural


BUILDDIR  := $(PROJECT)/protoc

PROTO_SRC_DIR := $(BUILDDIR)


Q ?= @
all: 
	@ echo PROTOC $<
	$(Q)protoc -I=$(PROTO_SRC_DIR) --python_out=$(BUILDDIR) $(PROTO_SRC_DIR)/$(PROJECT).proto
