# CUDA Configuration
# Override with: make COMPUTE_CAP=75 CUDA_HOME=/usr/local/cuda-11.8
COMPUTE_CAP ?= 61
CUDA_HOME ?= /usr/local/cuda-12.9
NVCC ?= $(CUDA_HOME)/bin/nvcc

# Compiler settings
CXXFLAGS = -O3 -std=c++11

# CUDA compiler flags
NVCCFLAGS = -O3 -std=c++11 \
	-gencode=arch=compute_$(COMPUTE_CAP),code=\"sm_$(COMPUTE_CAP),compute_$(COMPUTE_CAP)\" \
	-Xptxas="-v"

# Link flags
LDFLAGS = -lcurand

# Targets
TARGET = vanity_npub

# Default target
all: $(TARGET)

# Help target
help:
	@echo "CUDA Vanity Nostr npub Generator"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build vanity_npub (default)"
	@echo "  make info         - Show build configuration"
	@echo "  make detect       - Detect GPU and recommended settings"
	@echo "  make test         - Test with a sample pattern"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Configuration:"
	@echo "  COMPUTE_CAP       - GPU compute capability (default: 61 for GTX 1070)"
	@echo "  CUDA_HOME         - CUDA toolkit path (default: /usr/local/cuda-12.9)"
	@echo ""
	@echo "Examples:"
	@echo "  make COMPUTE_CAP=75 CUDA_HOME=/usr/local/cuda-11.8"
	@echo "  make info"
	@echo ""
	@echo "Usage:"
	@echo "  ./vanity_npub <pattern>"
	@echo "  ./vanity_npub test   # Find npub1test..."
	@echo ""
	@echo "Verify keys with nak:"
	@echo "  PRIVKEY=\$$(./vanity_npub test | grep 'Private key' | cut -d' ' -f3)"
	@echo "  nak key public \$$PRIVKEY | nak encode npub"
	@echo ""

# Detect GPU
detect:
	@echo "Detecting NVIDIA GPU..."
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | while IFS=',' read name cap; do \
			echo "  GPU: $$name"; \
			echo "  Compute Capability: $$cap"; \
			cap_int=$$(echo $$cap | tr -d '.'); \
			echo "  Recommended: make COMPUTE_CAP=$$cap_int"; \
		done; \
	else \
		echo "  nvidia-smi not found. Cannot detect GPU."; \
	fi
	@echo ""
	@echo "Available CUDA installations:"
	@ls -d /usr/local/cuda* 2>/dev/null | while read cuda_path; do \
		if [ -x "$$cuda_path/bin/nvcc" ]; then \
			version=$$($$cuda_path/bin/nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//'); \
			echo "  $$cuda_path (CUDA $$version)"; \
		fi; \
	done

# Display build configuration
info:
	@echo "Build Configuration:"
	@echo "  CUDA_HOME    = $(CUDA_HOME)"
	@echo "  NVCC         = $(NVCC)"
	@echo "  COMPUTE_CAP  = $(COMPUTE_CAP)"
	@echo "  NVCCFLAGS    = $(NVCCFLAGS)"
	@echo ""
	@echo "GPU Architecture: sm_$(COMPUTE_CAP)"
	@echo ""
	@echo "Common compute capabilities:"
	@echo "  35 - Tesla K40, Quadro K6000"
	@echo "  50 - GTX 9xx, Quadro M series"
	@echo "  52 - Tegra X1, GTX 9xx (Maxwell)"
	@echo "  61 - GTX 1050/1060/1070/1080, Titan X (Pascal)"
	@echo "  70 - Tesla V100, Titan V"
	@echo "  75 - GTX 16xx, RTX 20xx (Turing)"
	@echo "  80 - A100"
	@echo "  86 - RTX 30xx (Ampere)"
	@echo "  89 - RTX 40xx (Ada Lovelace)"
	@echo ""
	@echo "Override with: make COMPUTE_CAP=75 CUDA_HOME=/usr/local/cuda-11.8"

# Build target
$(TARGET): vanity_npub.cu secp256k1_jacobian.cuh
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) vanity_npub.cu $(LDFLAGS)

# Test target - run with sample pattern and verify with nak
test: $(TARGET)
	@echo "Running test to find npub1test..."
	@OUTPUT=$$(timeout 30 ./$(TARGET) test 2>&1); \
	echo "$$OUTPUT"; \
	PRIVKEY=$$(echo "$$OUTPUT" | grep "Private key" | awk '{print $$4}'); \
	if [ -n "$$PRIVKEY" ]; then \
		echo ""; \
		echo "Verifying with nak..."; \
		if command -v nak >/dev/null 2>&1; then \
			NPUB=$$(echo "$$PRIVKEY" | nak key public | nak encode npub); \
			echo "Verified npub: $$NPUB"; \
			if echo "$$NPUB" | grep -q "^npub1test"; then \
				echo "✓ Test PASSED - Key is valid and matches pattern!"; \
			else \
				echo "✗ Test FAILED - Key does not match pattern"; \
				exit 1; \
			fi; \
		else \
			echo "nak not found - skipping verification"; \
		fi; \
	else \
		echo "✗ Test FAILED - No key generated"; \
		exit 1; \
	fi

# Clean
clean:
	rm -f $(TARGET)

.PHONY: all info test clean help detect
