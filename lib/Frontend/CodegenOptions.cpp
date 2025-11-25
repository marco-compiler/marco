#include "marco/Frontend/CodegenOptions.h"

using namespace ::marco::frontend;

namespace marco::frontend {
CodegenOptions::CodegenOptions() : clang::CodeGenOptions() {
  vectorSizes.push_back(8);
}

bool CodegenOptions::hasFeature(llvm::StringRef feature) const {
  return llvm::find(features, feature) != features.end();
}

bool CodegenOptions::hasGPU() const { return gpuVendor != GPUVendor::None; }

GPUVendor CodegenOptions::getGPUVendor() const { return gpuVendor; }
} // namespace marco::frontend