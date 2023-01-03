#ifndef MARCO_CYCLESSYMBOLICSOLVER_H
#define MARCO_CYCLESSYMBOLICSOLVER_H

#include "mlir/IR/Builders.h"

namespace marco::codegen {
  class CyclesSymbolicSolver
  {
  private:
    mlir::OpBuilder& builder;

  public:
    CyclesSymbolicSolver(mlir::OpBuilder& builder);

    bool solve();

  };
}

#endif//MARCO_CYCLESSYMBOLICSOLVER_H
