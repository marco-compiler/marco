#ifndef MARCO_AST_MODELICA_ARRAYCONCATENATION_H
#define MARCO_AST_MODELICA_ARRAYCONCATENATION_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Array concatenation `[a, b; c, d]`.
class ArrayConcatenation
    : public NodeImpl<ArrayConcatenation, Expression,
                      NodeKind::Expression_ArrayConcatenation> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  size_t getNumRows() const;

  /// Get the elements of a row.
  ChildrenRef<Expression> getRow(size_t index) const;

  void addRow(Children<Expression> row);

private:
  llvm::SmallVector<Children<Expression>> rows;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ARRAYCONCATENATION_H
