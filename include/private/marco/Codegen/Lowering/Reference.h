#ifndef MARCO_CODEGEN_LOWERING_REFERENCE_H
#define MARCO_CODEGEN_LOWERING_REFERENCE_H

#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
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

      Reference& operator=(const Reference& other);

      Reference& operator=(Reference&& other);

      friend void swap(Reference& first, Reference& second);

      static Reference ssa(mlir::OpBuilder& builder, mlir::Value value);

      static Reference memory(mlir::OpBuilder& builder, mlir::Value value);

      static Reference variable(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::StringRef name,
          mlir::Type type);

      static Reference component(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::Value parent,
          mlir::Type componentType,
          llvm::StringRef componentName);

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
