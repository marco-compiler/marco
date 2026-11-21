#ifndef MARCO_AST_MODELICA_CLASSMODIFICATION_H
#define MARCO_AST_MODELICA_CLASSMODIFICATION_H

#include "marco/AST/Modelica/Argument.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Class modification: a parenthesized list of arguments. It is also used to
/// represent annotations.
class ClassModification
    : public NodeImpl<ClassModification, Node, NodeKind::ClassModification> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ChildrenRef<Argument> getArguments() const;

  void addArgument(std::unique_ptr<Argument> node);

private:
  Children<Argument> arguments;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_CLASSMODIFICATION_H
