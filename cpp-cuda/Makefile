# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -std=c++17 -O3

# Auto-detect all CUDA files in current directory
CUDA_SOURCES = $(wildcard *.cu)

# Generate target names from CUDA files (remove .cu extension)
TARGETS = $(CUDA_SOURCES:.cu=)

# Default target
all: $(TARGETS)

# Pattern rule: build any .cu file into an executable
%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Build all targets
build: $(TARGETS)

# Run all executables (for testing)
run: $(TARGETS)
	@for target in $(TARGETS); do \
		echo "=== Running $$target ==="; \
		./$$target; \
		echo ""; \
	done

# Clean all generated files
clean:
	rm -f $(TARGETS)

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build all .cu files (default)"
	@echo "  build     - Build all .cu files"
	@echo "  run       - Run all executables"
	@echo "  clean     - Remove all executables"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Available programs:"
	@for target in $(TARGETS); do \
		echo "  $$target - from $$target.cu"; \
	done
	@echo ""
	@echo "Usage examples:"
	@echo "  make                       # Build all"
	@echo "  make vector-addition       # Build specific program"
	@echo "  make clean                 # Clean all"
	@echo "  make run                   # Run all programs"

.PHONY: all build run clean help