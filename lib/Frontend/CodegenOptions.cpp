#include "marco/Frontend/CodegenOptions.h"

using namespace ::marco::frontend;

namespace marco::frontend {
bool CodegenOptions::hasGPU() const { return gpuVendor != GPUVendor::None; }

GPUVendor CodegenOptions::getGPUVendor() const { return gpuVendor; }
} // namespace marco::frontend
