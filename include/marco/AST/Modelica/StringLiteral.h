#ifndef MARCO_AST_MODELICA_STRINGLITERAL_H
#define MARCO_AST_MODELICA_STRINGLITERAL_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// String literal, with the escape sequences already resolved.
class StringLiteral : public NodeImpl<StringLiteral, Expression,
                                      NodeKind::Expression_StringLiteral> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  llvm::StringRef getValue() const;

  void setValue(std::string newValue);

private:
  std::string value;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_STRINGLITERAL_H
