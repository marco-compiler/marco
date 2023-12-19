#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

#include "llvm/Passes/OptimizationLevel.h"

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
    bool readOnlyVariablesPropagation = false;
    bool variablesToParametersPromotion = false;
    bool cse = false;
    bool omp = false;

    unsigned int bitWidth = 64;

    std::string target = "unknown";
    std::string cpu = "generic";
  };
}

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
