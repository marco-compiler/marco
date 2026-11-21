#ifndef MARCO_AST_MODELICA_INTEGERLITERAL_H
#define MARCO_AST_MODELICA_INTEGERLITERAL_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Integer literal, possibly with a unit.
class IntegerLiteral : public NodeImpl<IntegerLiteral, Expression,
                                       NodeKind::Expression_IntegerLiteral> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  int64_t getValue() const;

  void setValue(int64_t newValue);

  const std::optional<std::string> &getUnit() const;

  void setUnit(std::optional<std::string> newUnit);

private:
  int64_t value = 0;
  std::optional<std::string> unit;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_INTEGERLITERAL_H
