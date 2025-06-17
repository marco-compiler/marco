#ifndef MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H
#define MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include <functional>

namespace marco::codegen::lowering {
class OperationLowerer : public Lowerer {
public:
  using LoweringFunction = std::function<std::optional<Results>(
      OperationLowerer &, const ast::Operation &)>;

  explicit OperationLowerer(BridgeInterface *bridge);

  std::optional<Results> lower(const ast::Operation &operation) override;

protected:
  using Lowerer::lower;

private:
  std::optional<Results> negate(const ast::Operation &operation);
  std::optional<Results> add(const ast::Operation &operation);
  std::optional<Results> addEW(const ast::Operation &operation);
  std::optional<Results> subtract(const ast::Operation &operation);
  std::optional<Results> subtractEW(const ast::Operation &operation);
  std::optional<Results> multiply(const ast::Operation &operation);
  std::optional<Results> multiplyEW(const ast::Operation &operation);
  std::optional<Results> divide(const ast::Operation &operation);
  std::optional<Results> divideEW(const ast::Operation &operation);
  std::optional<Results> ifElse(const ast::Operation &operation);
  std::optional<Results> greater(const ast::Operation &operation);
  std::optional<Results> greaterOrEqual(const ast::Operation &operation);
  std::optional<Results> equal(const ast::Operation &operation);
  std::optional<Results> notEqual(const ast::Operation &operation);
  std::optional<Results> lessOrEqual(const ast::Operation &operation);
  std::optional<Results> less(const ast::Operation &operation);
  std::optional<Results> logicalAnd(const ast::Operation &operation);
  std::optional<Results> logicalNot(const ast::Operation &operation);
  std::optional<Results> logicalOr(const ast::Operation &operation);
  std::optional<Results> subscription(const ast::Operation &operation);
  std::optional<Results> powerOf(const ast::Operation &operation);
  std::optional<Results> powerOfEW(const ast::Operation &operation);
  std::optional<Results> range(const ast::Operation &operation);

private:
  std::optional<mlir::Value> lowerArg(const ast::Expression &expression);

  [[nodiscard]] virtual bool
  lowerArgs(const ast::Operation &operation,
            llvm::SmallVectorImpl<mlir::Value> &args);
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_OPERATIONLOWERER_H
