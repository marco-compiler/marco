#ifndef MARCO_AST_MODELICA_IFEXPRESSION_H
#define MARCO_AST_MODELICA_IFEXPRESSION_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// If-expression. The `elseif` branches are kept as
/// separate conditions.
class IfExpression
    : public NodeImpl<IfExpression, Expression, NodeKind::Expression_If> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  /// Conditions of the if and elseif branches.
  ChildrenRef<Expression> getConditions() const;

  /// Values of the if and elseif branches.
  ChildrenRef<Expression> getValues() const;

  void addBranch(std::unique_ptr<Expression> condition,
                 std::unique_ptr<Expression> value);

  Expression *getElseValue() const;

  void setElseValue(std::unique_ptr<Expression> node);

private:
  Children<Expression> conditions;
  Children<Expression> values;
  Child<Expression> elseValue;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_IFEXPRESSION_H
