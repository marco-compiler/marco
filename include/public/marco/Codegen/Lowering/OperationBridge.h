#ifndef MARCO_CODEGEN_LOWERING_OPERATIONBRIDGE_H
#define MARCO_CODEGEN_LOWERING_OPERATIONBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/NewBridge.h"
#include <functional>

namespace marco::codegen::lowering
{
  class OperationBridge
  {
    public:
      using Lowerer = std::function<Results(OperationBridge&, const ast::Operation&)>;

      OperationBridge(NewLoweringBridge* bridge);

      Results negate(const ast::Operation& operation);
      Results add(const ast::Operation& operation);
      Results subtract(const ast::Operation& operation);
      Results multiply(const ast::Operation& operation);
      Results divide(const ast::Operation& operation);
      Results ifElse(const ast::Operation& operation);
      Results greater(const ast::Operation& operation);
      Results greaterOrEqual(const ast::Operation& operation);
      Results equal(const ast::Operation& operation);
      Results notEqual(const ast::Operation& operation);
      Results lessOrEqual(const ast::Operation& operation);
      Results less(const ast::Operation& operation);
      Results logicalAnd(const ast::Operation& operation);
      Results logicalOr(const ast::Operation& operation);
      Results subscription(const ast::Operation& operation);
      Results memberLookup(const ast::Operation& operation);
      Results powerOf(const ast::Operation& operation);

    private:
      mlir::OpBuilder& builder();

      std::vector<mlir::Value> lowerArgs(const ast::Operation& operation);

      template<ast::OperationKind Kind, int Arity = -1>
      Results lowerOperation(const ast::Operation& operation, std::function<Results(mlir::Location, mlir::ValueRange)> callback)
      {
        assert(operation.getOperationKind() == Kind);

        auto args = lowerArgs(operation);
        assert(Arity == -1 || args.size() == Arity);

        mlir::Location loc = bridge->loc(operation.getLocation());
        return callback(std::move(loc), args);
      }

      NewLoweringBridge* bridge;
  };
}

#endif // MARCO_CODEGEN_LOWERING_OPERATIONBRIDGE_H
