#ifndef MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include <functional>

namespace marco::codegen::lowering {
class OperationLowerer : public Lowerer {
public:
  using LoweringFunction = std::function<std::optional<Results>(
      OperationLowerer &, const ast::bmodelica::Operation &)>;

  explicit OperationLowerer(BridgeInterface *bridge);

  std::optional<Results>
  lower(const ast::bmodelica::Operation &operation) override;

protected:
  using Lowerer::lower;

private:
  std::optional<Results> negate(const ast::bmodelica::Operation &operation);
  std::optional<Results> add(const ast::bmodelica::Operation &operation);
  std::optional<Results> addEW(const ast::bmodelica::Operation &operation);
  std::optional<Results> subtract(const ast::bmodelica::Operation &operation);
  std::optional<Results> subtractEW(const ast::bmodelica::Operation &operation);
  std::optional<Results> multiply(const ast::bmodelica::Operation &operation);
  std::optional<Results> multiplyEW(const ast::bmodelica::Operation &operation);
  std::optional<Results> divide(const ast::bmodelica::Operation &operation);
  std::optional<Results> divideEW(const ast::bmodelica::Operation &operation);
  std::optional<Results> ifElse(const ast::bmodelica::Operation &operation);
  std::optional<Results> greater(const ast::bmodelica::Operation &operation);
  std::optional<Results>
  greaterOrEqual(const ast::bmodelica::Operation &operation);
  std::optional<Results> equal(const ast::bmodelica::Operation &operation);
  std::optional<Results> notEqual(const ast::bmodelica::Operation &operation);
  std::optional<Results>
  lessOrEqual(const ast::bmodelica::Operation &operation);
  std::optional<Results> less(const ast::bmodelica::Operation &operation);
  std::optional<Results> logicalAnd(const ast::bmodelica::Operation &operation);
  std::optional<Results> logicalNot(const ast::bmodelica::Operation &operation);
  std::optional<Results> logicalOr(const ast::bmodelica::Operation &operation);
  std::optional<Results>
  subscription(const ast::bmodelica::Operation &operation);
  std::optional<Results> powerOf(const ast::bmodelica::Operation &operation);
  std::optional<Results> powerOfEW(const ast::bmodelica::Operation &operation);
  std::optional<Results> range(const ast::bmodelica::Operation &operation);

private:
  std::optional<mlir::Value>
  lowerArg(const ast::bmodelica::Expression &expression);

  [[nodiscard]] virtual bool
  lowerArgs(const ast::bmodelica::Operation &operation,
            llvm::SmallVectorImpl<mlir::Value> &args);
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H
