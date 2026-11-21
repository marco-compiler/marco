#ifndef MARCO_AST_MODELICA_OUTPUTLIST_H
#define MARCO_AST_MODELICA_OUTPUTLIST_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Parenthesized list of expressions with possibly omitted elements, as in
/// `(a, , b) = f(x)`. A single parenthesized expression is not represented
/// by this node.
class OutputList
    : public NodeImpl<OutputList, Expression, NodeKind::Expression_OutputList> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  /// Get the elements. Omitted elements are null.
  ChildrenRef<Expression> getElements() const;

  void addElement(std::unique_ptr<Expression> node);

private:
  Children<Expression> elements;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_OUTPUTLIST_H
