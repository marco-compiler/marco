#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

#include "clang/Basic/CodeGenOptions.h"
#include "llvm/Passes/OptimizationLevel.h"
#include <string>
#include <vector>

namespace marco::frontend {
/// Code generation options.
/// The default values are for compiling without optimizations.
/// The class extends the language options for C / C++ to enable the
/// integration with clang's diagnostics infrastructure.
struct CodegenOptions : public clang::CodeGenOptions {
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
  bool runtimeVerification = true;

  bool dataRecomputation = false;
  bool drCostModel = false;
  std::string drCpuCostModelFile;
  bool drSummary = false;
  bool drDebug = false;
  bool drPartialRemat = false;
  bool drBufferElim = false;
  bool drEraseEliminatedBuffers = false;
  bool drFootprintAnalysis = false;
  unsigned drPartialMaxLeaves = 4;
  unsigned drL1Size = 32768;
  unsigned drL2Size = 262144;
  unsigned drL3Size = 0;
  unsigned drL1Latency = 4;
  unsigned drL2Latency = 12;
  unsigned drL3Latency = 40;
  unsigned drMemLatency = 200;
  unsigned drCacheLineSize = 64;
  unsigned drRegBudget = 32;
  unsigned drSpillCycles = 4;
  unsigned drIcacheSoftBudget = 128;

  uint64_t bitWidth = 64;

  std::string cpu = "generic";
  std::vector<std::string> features;
};
} // namespace marco::frontend

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
