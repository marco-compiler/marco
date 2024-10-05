#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

#include "clang/Basic/CodeGenOptions.h"
#include "llvm/Passes/OptimizationLevel.h"
#include <string>
#include <vector>

namespace marco::frontend
{
  /// Code generation operations.
  /// The default values are for compiling without optimizations.
  /// The class extends the language options for C / C++ to enable the
  /// integration with clang's diagnostics infrastructure.
  struct CodegenOptions : public clang::CodeGenOptions
  {
    llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0;

    bool debug = true;
    bool assertions = true;
    bool inlining = false;
    bool outputArraysPromotion = false;
    bool heapToStackPromotion = false;
    bool readOnlyVariablesPropagation = false;
    bool variablesToParametersPromotion = false;
    bool cse = false;
    bool equationsRuntimeScheduling = false;
    bool omp = false;
    bool singleValuedInductionElimination = false;
    bool loopFusion = false;
    bool loopCoalescing = false;
    bool loopTiling = false;

    unsigned int bitWidth = 64;

    std::string cpu = "generic";
    std::vector<std::string> features;
  };
}

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
