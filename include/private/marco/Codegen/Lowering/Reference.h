#ifndef MARCO_CODEGEN_LOWERING_REFERENCE_H
#define MARCO_CODEGEN_LOWERING_REFERENCE_H

#include "mlir/IR/Builders.h"

namespace marco::codegen::lowering
{
  class Reference
  {
    public:
      Reference();

      mlir::Value operator*();
      mlir::Value getReference() const;

      void set(mlir::Value value);

      static Reference ssa(mlir::OpBuilder* builder, mlir::Value value);
      static Reference memory(mlir::OpBuilder* builder, mlir::Value value);
      static Reference member(mlir::OpBuilder* builder, mlir::Value value);
      static Reference time(mlir::OpBuilder* builder);

    private:
      Reference(mlir::OpBuilder* builder,
                mlir::Value value,
                std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader,
                std::function<void(mlir::OpBuilder* builder, Reference& destination, mlir::Value)> writer);

      mlir::OpBuilder* builder;
      mlir::Value value;
      std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader;
      std::function<void(mlir::OpBuilder*, Reference&, mlir::Value)> writer;
  };
}

#endif // MARCO_CODEGEN_LOWERING_REFERENCE_H
