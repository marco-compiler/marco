#ifndef MARCO_TEST_MODELING_COMMON_H
#define MARCO_TEST_MODELING_COMMON_H

#include "mlir/IR/AffineMap.h"

namespace marco::test
{
  mlir::AffineExpr getDimWithOffset(
      unsigned int dimension, int64_t offset, mlir::MLIRContext* context);
}

#endif // MARCO_TEST_MODELING_COMMON_H
