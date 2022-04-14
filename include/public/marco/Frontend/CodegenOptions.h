#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

namespace marco::frontend
{
  /// Optimization level
  struct OptLevel
  {
    unsigned int time = 0;
    unsigned int size = 0;
  };

  /// Code generation operations.
  /// The default values are for compiling without optimizations.
  struct CodegenOptions
  {
    OptLevel optLevel;

    bool debug = true;
    bool assertions = true;
    bool inlining = false;
    bool outputArraysPromotion = false;
    bool cse = false;
    bool omp = false;

    unsigned int bitWidth = 64;
    bool generateMain = true;
    bool cWrappers = false;
  };
}

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
