#ifndef MARCO_AST_MODELICA_BOOLEANLITERAL_H
#define MARCO_AST_MODELICA_BOOLEANLITERAL_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
class BooleanLiteral : public NodeImpl<BooleanLiteral, Expression,
                                       NodeKind::Expression_BooleanLiteral> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  bool getValue() const;

  void setValue(bool newValue);

private:
  bool value = false;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_BOOLEANLITERAL_H
