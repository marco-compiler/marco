#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_BRIDGE_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_BRIDGE_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/Results.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>

namespace marco::codegen::lowering::bmodelica {
class Bridge {
private:
  class Impl;
  std::unique_ptr<Impl> impl;

public:
  Bridge(mlir::MLIRContext &context);

  ~Bridge();

  [[nodiscard]] bool lower(const ast::bmodelica::Root &root);

  std::unique_ptr<mlir::ModuleOp> &getMLIRModule();
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_BRIDGE_H
