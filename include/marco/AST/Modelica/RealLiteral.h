#ifndef MARCO_AST_MODELICA_REALLITERAL_H
#define MARCO_AST_MODELICA_REALLITERAL_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Real literal, possibly with a unit.
class RealLiteral
    : public NodeImpl<RealLiteral, Expression, NodeKind::Expression_RealLiteral> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  double getValue() const;

  void setValue(double newValue);

  const std::optional<std::string> &getUnit() const;

  void setUnit(std::optional<std::string> newUnit);

private:
  double value = 0;
  std::optional<std::string> unit;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_REALLITERAL_H
