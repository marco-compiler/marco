#ifndef MARCO_AST_MODELICA_CALL_H
#define MARCO_AST_MODELICA_CALL_H

#include "marco/AST/Modelica/ComponentReference.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/ForIndex.h"
#include "marco/AST/Modelica/NamedArgument.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Function call. The callee is a component reference,
/// whose single part is `der`, `initial` or `pure` for the homonymous
/// operators. The arguments are either positional and named arguments, or a
/// single expression followed by iterators (e.g., `sum(i for i in 1:n)`).
class Call : public NodeImpl<Call, Expression, NodeKind::Expression_Call> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ComponentReference *getCallee() const;

  void setCallee(std::unique_ptr<ComponentReference> node);

  ChildrenRef<Expression> getArguments() const;

  void addArgument(std::unique_ptr<Expression> node);

  ChildrenRef<NamedArgument> getNamedArguments() const;

  void addNamedArgument(std::unique_ptr<NamedArgument> node);

  /// Iterators of reductions and array constructors with iterators. If
  /// present, the call has exactly one positional argument.
  ChildrenRef<ForIndex> getIterators() const;

  void addIterator(std::unique_ptr<ForIndex> node);

private:
  Child<ComponentReference> callee;
  Children<Expression> arguments;
  Children<NamedArgument> namedArguments;
  Children<ForIndex> iterators;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_CALL_H
