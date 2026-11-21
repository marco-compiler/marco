#ifndef MARCO_AST_MODELICA_EQUALITYEQUATION_H
#define MARCO_AST_MODELICA_EQUALITYEQUATION_H

#include "marco/AST/Modelica/Equation.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Equation `lhs = rhs`. The left-hand side may be an
/// OutputList, as in `(a, b) = f(x)`.
class EqualityEquation : public NodeImpl<EqualityEquation, Equation,
                                         NodeKind::Equation_Equality> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  Expression *getLhs() const;

  void setLhs(std::unique_ptr<Expression> node);

  Expression *getRhs() const;

  void setRhs(std::unique_ptr<Expression> node);

private:
  Child<Expression> lhs;
  Child<Expression> rhs;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_EQUALITYEQUATION_H
