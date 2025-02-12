#include "Common.h"

namespace marco::test {
mlir::AffineExpr getDimWithOffset(unsigned int dimension, int64_t offset,
                                  mlir::MLIRContext *context) {
  mlir::AffineExpr dimExpr = mlir::getAffineDimExpr(dimension, context);
  mlir::AffineExpr offsetExpr = mlir::getAffineConstantExpr(offset, context);
  return dimExpr + offsetExpr;
}
} // namespace marco::test
