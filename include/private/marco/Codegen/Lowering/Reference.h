#ifndef MARCO_CODEGEN_LOWERING_REFERENCE_H
#define MARCO_CODEGEN_LOWERING_REFERENCE_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/Builders.h"
#include <memory>

namespace marco::codegen::lowering
{
  class Reference
  {
    public:
      class Impl;

    public:
      Reference();

      ~Reference();

      Reference(const Reference& other);

      static Reference ssa(mlir::OpBuilder& builder, mlir::Value value);

      static Reference memory(mlir::OpBuilder& builder, mlir::Value value);

      static Reference variable(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::StringRef name,
          mlir::Type type);

      static Reference time(mlir::OpBuilder& builder, mlir::Location loc);

      mlir::Location getLoc() const;

      mlir::Value getReference() const;

      mlir::Value get(mlir::Location loc) const;

      void set(mlir::Location loc, mlir::Value value);

    private:
      Reference(std::unique_ptr<Impl> impl);

    private:
      std::unique_ptr<Impl> impl;
  };
}

#endif // MARCO_CODEGEN_LOWERING_REFERENCE_H
