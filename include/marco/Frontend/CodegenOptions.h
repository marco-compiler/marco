#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

#include "clang/Basic/CodeGenOptions.h"
#include "llvm/Passes/OptimizationLevel.h"
#include <string>
#include <vector>

namespace marco::frontend {
enum class GPUVendor {
  None,
  AMD,
  Intel,
  NVIDIA
};

/// Code generation options.
/// The default values are for compiling without optimizations.
/// The class extends the language options for C / C++ to enable the
/// integration with clang's diagnostics infrastructure.
struct CodegenOptions : public clang::CodeGenOptions {
  CodegenOptions();

  llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0;

  bool debug = true;
  bool assertions = true;
  bool inlining = false;
  bool outputArraysPromotion = false;
  bool heapToStackPromotion = false;
  bool mem2reg = false;
  bool readOnlyVariablesPropagation = false;
  bool matchingGraphScalarization = true;
  double matchingGraphScalarizationThreshold = 0.5;
  bool variablesPruning = false;
  bool variablesToParametersPromotion = false;
  int64_t sccSolvingBySubstitutionMaxIterations = 100;
  int64_t sccSolvingBySubstitutionMaxEquationsInSCC = 5;
  bool cse = false;
  bool functionCallsCSE = false;
  bool equationsRuntimeScheduling = false;
  bool omp = false;
  bool singleValuedInductionElimination = false;
  bool loopHoisting = false;
  bool loopTiling = false;
  bool vectorization = false;
  llvm::SmallVector<int64_t> vectorSizes;
  bool runtimeVerification = true;

  uint64_t bitWidth = 64;

  std::string cpu = "generic";
  std::vector<std::string> features;

  // TODO default to empty and detect at runtime
  std::string gpuTriple = "nvptx64-nvidia-cuda";
  GPUVendor gpuVendor = GPUVendor::NVIDIA;
  std::string gpuChip = "sm_86";
  std::string gpuFeatures = "+ptx60";

  bool hasFeature(llvm::StringRef feature) const;

  bool hasGPU() const;

  GPUVendor getGPUVendor() const;
};
} // namespace marco::frontend

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
