#ifndef MARCO_AST_MODELICA_SUBSCRIPTED_H
#define MARCO_AST_MODELICA_SUBSCRIPTED_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Subscripts applied to a parenthesized expression, as in `(a * b)[1, :]`.
class Subscripted : public NodeImpl<Subscripted, Expression,
                                    NodeKind::Expression_Subscripted> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  Expression *getBase() const;

  void setBase(std::unique_ptr<Expression> node);

  ChildrenRef<Expression> getSubscripts() const;

  void addSubscript(std::unique_ptr<Expression> node);

private:
  Child<Expression> base;
  Children<Expression> subscripts;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_SUBSCRIPTED_H
