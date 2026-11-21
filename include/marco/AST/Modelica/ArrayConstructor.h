#ifndef MARCO_AST_MODELICA_ARRAYCONSTRUCTOR_H
#define MARCO_AST_MODELICA_ARRAYCONSTRUCTOR_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/ForIndex.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Array constructor `{a, b, c}` or with iterators `{e for i in r}`.
class ArrayConstructor
    : public NodeImpl<ArrayConstructor, Expression,
                      NodeKind::Expression_ArrayConstructor> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ChildrenRef<Expression> getElements() const;

  void addElement(std::unique_ptr<Expression> node);

  /// Iterators of the array constructor. If present, the constructor has
  /// exactly one element.
  ChildrenRef<ForIndex> getIterators() const;

  void addIterator(std::unique_ptr<ForIndex> node);

private:
  Children<Expression> elements;
  Children<ForIndex> iterators;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ARRAYCONSTRUCTOR_H
