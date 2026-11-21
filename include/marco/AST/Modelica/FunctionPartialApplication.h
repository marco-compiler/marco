#ifndef MARCO_AST_MODELICA_FUNCTIONPARTIALAPPLICATION_H
#define MARCO_AST_MODELICA_FUNCTIONPARTIALAPPLICATION_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NamedArgument.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Function partial application, usable only as
/// function argument.
class FunctionPartialApplication
    : public NodeImpl<FunctionPartialApplication, Expression,
                      NodeKind::Expression_FunctionPartialApplication> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  const Name &getFunction() const;

  void setFunction(Name name);

  ChildrenRef<NamedArgument> getNamedArguments() const;

  void addNamedArgument(std::unique_ptr<NamedArgument> node);

private:
  Name function;
  Children<NamedArgument> namedArguments;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_FUNCTIONPARTIALAPPLICATION_H
