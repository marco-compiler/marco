#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

namespace marco::frontend
{
  struct OptLevel
  {
    unsigned int time = 2;
    unsigned int size = 0;
  };

  struct CodegenOptions
  {
    bool debug = false;
    bool assertions = true;

    bool generateMain = true;
    bool inlining = true;
    bool outputArraysPromotion = true;
    bool cse = true;
    bool omp = false;
    bool cWrappers = false;

    OptLevel optLevel;
  };
}

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
