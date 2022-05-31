#ifndef MARCO_CODEGEN_LOWERING_OPERATIONBRIDGE_H
#define MARCO_CODEGEN_LOWERING_OPERATIONBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"
#include <functional>

namespace marco::codegen::lowering
{
  class OperationLowerer : public Lowerer
  {
    public:
      using LoweringFunction = std::function<Results(OperationLowerer&, const ast::Operation&)>;

      OperationLowerer(LoweringContext* context, BridgeInterface* bridge);

      Results negate(const ast::Operation& operation);
      Results add(const ast::Operation& operation);
      Results addEW(const ast::Operation& operation);
      Results subtract(const ast::Operation& operation);
      Results subtractEW(const ast::Operation& operation);
      Results multiply(const ast::Operation& operation);
      Results multiplyEW(const ast::Operation& operation);
      Results divide(const ast::Operation& operation);
      Results divideEW(const ast::Operation& operation);
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
      Results powerOfEW(const ast::Operation& operation);

    protected:
      using Lowerer::lower;

    private:
      std::vector<mlir::Value> lowerArgs(const ast::Operation& operation);

      template<ast::OperationKind Kind>
      Results lowerOperation(const ast::Operation& operation, std::function<Results(mlir::Location, mlir::ValueRange)> callback)
      {
        assert(operation.getOperationKind() == Kind);
        auto args = lowerArgs(operation);
        auto location = loc(operation.getLocation());
        return callback(std::move(location), args);
      }
  };
}

#endif // MARCO_CODEGEN_LOWERING_OPERATIONBRIDGE_H
