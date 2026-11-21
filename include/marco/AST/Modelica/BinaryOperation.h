#ifndef MARCO_AST_MODELICA_BINARYOPERATION_H
#define MARCO_AST_MODELICA_BINARYOPERATION_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
enum class BinaryOperator {
  Add,
  AddEW,
  Sub,
  SubEW,
  Mul,
  MulEW,
  Div,
  DivEW,
  Pow,
  PowEW,
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  And,
  Or
};

llvm::StringRef toString(BinaryOperator op);

class BinaryOperation : public NodeImpl<BinaryOperation, Expression,
                                        NodeKind::Expression_BinaryOperation> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  BinaryOperator getOperator() const;

  void setOperator(BinaryOperator newOp);

  Expression *getLhs() const;

  void setLhs(std::unique_ptr<Expression> node);

  Expression *getRhs() const;

  void setRhs(std::unique_ptr<Expression> node);

private:
  BinaryOperator op = BinaryOperator::Add;
  Child<Expression> lhs;
  Child<Expression> rhs;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_BINARYOPERATION_H
