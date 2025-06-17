#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_IDA_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_IDA_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_IDAPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createIDAPass();

std::unique_ptr<mlir::Pass> createIDAPass(const IDAPassOptions &options);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_IDA_H
