#ifndef MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H

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

      OperationLowerer(BridgeInterface* bridge);

      Results lower(const ast::Operation& operation) override;

    protected:
      using Lowerer::lower;

    private:
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
      Results logicalNot(const ast::Operation& operation);
      Results logicalOr(const ast::Operation& operation);
      Results subscription(const ast::Operation& operation);
      Results powerOf(const ast::Operation& operation);
      Results powerOfEW(const ast::Operation& operation);

      template<ast::OperationKind OperationKind>
      bool inferResultTypes(
          mlir::MLIRContext* context,
          llvm::ArrayRef<mlir::Value> operands,
          llvm::SmallVectorImpl<mlir::Type>& inferredTypes);

    private:
      mlir::Value lowerArg(const ast::Expression& expression);

      void lowerArgs(
          const ast::Operation& operation,
          llvm::SmallVectorImpl<mlir::Value>& args);
  };
}

#endif // MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H
