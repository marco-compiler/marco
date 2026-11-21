#ifndef MARCO_AST_MODELICA_NAMEDARGUMENT_H
#define MARCO_AST_MODELICA_NAMEDARGUMENT_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Named argument of a function call (`name = value`).
class NamedArgument
    : public NodeImpl<NamedArgument, Node, NodeKind::NamedArgument> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  llvm::StringRef getName() const;

  void setName(std::string newName);

  Expression *getValue() const;

  void setValue(std::unique_ptr<Expression> node);

private:
  std::string name;
  Child<Expression> value;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_NAMEDARGUMENT_H
