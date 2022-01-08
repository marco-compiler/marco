#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

namespace marco::frontend
{
  struct CodegenOptions
  {
    bool debug;
    bool assertions;

    bool generateMain;
    bool inlining;
    bool outputArraysPromotion;
    bool cse;
    bool omp;
    bool cWrappers;
  };
}

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
