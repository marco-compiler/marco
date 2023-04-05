#ifndef MARCO_CODEGEN_BRIDGE_H
#define MARCO_CODEGEN_BRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Results.h"
#include "marco/Codegen/Options.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>

namespace marco::codegen::lowering
{
  class Bridge
  {
    private:
      class Impl;
      std::unique_ptr<Impl> impl;

    public:
      Bridge(
        mlir::MLIRContext& context,
        CodegenOptions options = CodegenOptions::getDefaultOptions());

      ~Bridge();

      void lower(const ast::Root& root);

      std::unique_ptr<mlir::ModuleOp>& getMLIRModule();
  };
}

#endif // MARCO_CODEGEN_BRIDGE_H
