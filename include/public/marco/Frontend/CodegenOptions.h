#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

#include "llvm/Passes/OptimizationLevel.h"
#include <string>

namespace marco::frontend
{
  /// Code generation operations.
  /// The default values are for compiling without optimizations.
  struct CodegenOptions
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
    bool loopFusion = false;
    bool loopCoalescing = false;
    bool loopTiling = false;

    unsigned int bitWidth = 64;

    std::string target = "unknown";
    std::string cpu = "generic";
  };
}

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
