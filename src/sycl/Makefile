TARGET_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_NAME := $(notdir $(TARGET_DIR))
TARGET := $(BASE_DIR)/$(TARGET_NAME)
include Makefile.deps
EXTERNAL_DEPENDS := $(sycl_EXTERNAL_DEPENDS)

$(TARGET):
test_cpu:
test_nvidiagpu:
test_intelagpu:
test_auto: $(TARGET)
	@echo
	@echo "Testing $(TARGET)"
	$(TARGET) --maxEvents 2
	@echo "Succeeded"
.PHONY: test_cpu test_nvidiagpu test_intelgpu test_auto

EXE_SRC := $(wildcard $(TARGET_DIR)/bin/*.cc)
EXE_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(EXE_SRC:%=%.o))
EXE_DEP := $(EXE_OBJ:$.o=$.d)

LIBNAMES := $(filter-out plugin-% bin test Makefile% plugins.txt%,$(wildcard *))
PLUGINNAMES := $(patsubst plugin-%,%,$(filter plugin-%,$(wildcard *)))
MY_CXXFLAGS := -I$(TARGET_DIR) -DLIB_DIR=$(LIB_DIR)/$(TARGET_NAME)
MY_LDFLAGS := -ldl -Wl,-rpath,$(LIB_DIR)/$(TARGET_NAME)
LIB_LDFLAGS := -L$(LIB_DIR)/$(TARGET_NAME)

ALL_DEPENDS := $(EXE_DEP)
# Files for libraries
LIBS :=
define LIB_template
$(1)_SRC := $$(wildcard $(TARGET_DIR)/$(1)/*.cc)
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
ALL_DEPENDS += $$($(1)_DEP)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/lib$(1).so
LIBS += $$($(1)_LIB)
$(1)_LDFLAGS := -l$(1)
endef
$(foreach lib,$(LIBNAMES),$(eval $(call LIB_template,$(lib))))

# Files for plugins
PLUGINS :=
define PLUGIN_template
$(1)_SRC := $$(wildcard $(TARGET_DIR)/plugin-$(1)/*.cc)
$(1)_OBJ := $$(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$$($(1)_SRC:%=%.o))
$(1)_DEP := $$($(1)_OBJ:$.o=$.d)
ALL_DEPENDS += $$($(1)_DEP)
$(1)_LIB := $(LIB_DIR)/$(TARGET_NAME)/plugin$(1).so
PLUGINS += $$($(1)_LIB)
endef
$(foreach lib,$(PLUGINNAMES),$(eval $(call PLUGIN_template,$(lib))))

# Files for unit tests
TESTS_SRC := $(wildcard $(TARGET_DIR)/test/*.cc)
TESTS_OBJ := $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(TESTS_SRC:%=%.o))
TESTS_DEP := $(TESTS_OBJ:$.o=$.d)
TESTS_EXE_CPU := $(patsubst $(SRC_DIR)/$(TARGET_NAME)/test/%.cc,$(TEST_DIR)/$(TARGET_NAME)/%,$(TESTS_SRC))
TESTS_EXE := $(TESTS_EXE_CPU)
ALL_DEPENDS += $(TESTS_DEP)
# Needed to keep the unit test object files after building $(TARGET)
.SECONDARY: $(TESTS_OBJ)

define RUNTEST_template
run_$(1): $(1)
	@echo
	@echo "Running test $(1)"
	@$(1)
	@echo "Succeeded"
test_$(2): run_$(1)
endef
$(foreach test,$(TESTS_EXE_CPU),$(eval $(call RUNTEST_template,$(test),auto)))

-include $(ALL_DEPENDS)

# Build targets
$(LIB_DIR)/$(TARGET_NAME)/plugins.txt: $(PLUGINS)
	nm -A -C -D -P --defined-only $(PLUGINS) | sed -n -e"s#$(LIB_DIR)/$(TARGET_NAME)/\(plugin\w\+\.so\): typeinfo for edm::\(PluginFactory\|ESPluginFactory\)::impl::Maker<\([A-Za-z0-9_:]\+\)> V .* .*#\3 \1#p" | sort > $@

$(TARGET): $(EXE_OBJ) $(LIBS) $(PLUGINS) $(LIB_DIR)/$(TARGET_NAME)/plugins.txt | $(TESTS_EXE)
	$(SYCL_CXX) $(SYCL_CXXFLAGS) $(EXE_OBJ) $(LDFLAGS) $(MY_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))

define BUILD_template
$(OBJ_DIR)/$(2)/%.cc.o: $(SRC_DIR)/$(2)/%.cc
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(SYCL_CXX) $(SYCL_CXXFLAGS) $(MY_CXXFLAGS) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_CXXFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_SYCL_CXXFLAGS)) -c $$< -o $$@ -MMD
	@cp $(OBJ_DIR)/$(2)/$$*.cc.d $(OBJ_DIR)/$(2)/$$*.cc.d.tmp; \
	  sed 's#\($(2)/$$*\)\.o[ :]*#\1.o \1.d : #g' < $(OBJ_DIR)/$(2)/$$*.cc.d.tmp > $(OBJ_DIR)/$(2)/$$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$$$//' \
	      -e '/^$$$$/ d' -e 's/$$$$/ :/' -e 's/ *//' < $(OBJ_DIR)/$(2)/$$*.cc.d.tmp >> $(OBJ_DIR)/$(2)/$$*.cc.d; \
	  rm $(OBJ_DIR)/$(2)/$$*.cc.d.tmp

$$($(1)_LIB): $$($(1)_OBJ) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LIB))
	@[ -d $$(@D) ] || mkdir -p $$(@D)
	$(SYCL_CXX) $(SYCL_CXXFLAGS) $$($(1)_OBJ) $(LDFLAGS) -shared $(SO_LDFLAGS) $(LIB_LDFLAGS) $$(foreach lib,$$($(1)_DEPENDS),$$($$(lib)_LDFLAGS)) $$(foreach dep,$(EXTERNAL_DEPENDS),$$($$(dep)_LDFLAGS)) -o $$@
endef

$(foreach lib,$(LIBNAMES),$(eval $(call BUILD_template,$(lib),$(TARGET_NAME)/$(lib))))
$(foreach lib,$(PLUGINNAMES),$(eval $(call BUILD_template,$(lib),$(TARGET_NAME)/plugin-$(lib))))

$(OBJ_DIR)/$(TARGET_NAME)/bin/%.cc.o: $(SRC_DIR)/$(TARGET_NAME)/bin/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(SYCL_CXX) $(SYCL_CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_SYCL_CXXFLAGS)) -c $< -o $@ -MMD
	@cp $(@D)/$*.cc.d $(@D)/$*.cc.d.tmp; \
	  sed 's#\($(TARGET_NAME)/$*\)\.o[ :]*#\1.o \1.d : #g' < $(@D)/$*.cc.d.tmp > $(@D)/$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' -e 's/ *//' < $(@D)/$*.cc.d.tmp >> $(@D)/$*.cc.d; \
	  rm $(@D)/$*.cc.d.tmp

# Tests
$(OBJ_DIR)/$(TARGET_NAME)/test/%.cc.o: $(SRC_DIR)/$(TARGET_NAME)/test/%.cc
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(SYCL_CXX) $(SYCL_CXXFLAGS) $(MY_CXXFLAGS) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_CXXFLAGS)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_SYCL_CXXFLAGS)) -c $< -o $@ -MMD
	@cp $(@D)/$*.cc.d $(@D)/$*.cc.d.tmp; \
	  sed 's#\($(TARGET_NAME)/$*\)\.o[ :]*#\1.o \1.d : #g' < $(@D)/$*.cc.d.tmp > $(@D)/$*.cc.d; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' -e 's/ *//' < $(@D)/$*.cc.d.tmp >> $(@D)/$*.cc.d; \
	  rm $(@D)/$*.cc.d.tmp

$(TEST_DIR)/$(TARGET_NAME)/%: $(OBJ_DIR)/$(TARGET_NAME)/test/%.cc.o | $(LIBS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(SYCL_CXX) $(SYCL_CXXFLAGS) $< $(LDFLAGS) $(MY_LDFLAGS) -o $@ -L$(LIB_DIR)/$(TARGET_NAME) $(patsubst %,-l%,$(LIBNAMES)) $(foreach dep,$(EXTERNAL_DEPENDS),$($(dep)_LDFLAGS))
