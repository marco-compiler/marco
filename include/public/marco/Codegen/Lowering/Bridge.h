#ifndef MARCO_CODEGEN_LOWERING_BRIDGE_H
#define MARCO_CODEGEN_LOWERING_BRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>

namespace marco::codegen::lowering
{
  class Bridge
  {
    private:
      class Impl;
      std::unique_ptr<Impl> impl;

    public:
      Bridge(mlir::MLIRContext& context);

      ~Bridge();

      void lower(const ast::Root& root);

      std::unique_ptr<mlir::ModuleOp>& getMLIRModule();
  };
}

#endif // MARCO_CODEGEN_LOWERING_BRIDGE_H
