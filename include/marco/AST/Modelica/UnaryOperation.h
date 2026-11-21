#ifndef MARCO_AST_MODELICA_UNARYOPERATION_H
#define MARCO_AST_MODELICA_UNARYOPERATION_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
enum class UnaryOperator { Plus, Minus, PlusEW, MinusEW, Not };

llvm::StringRef toString(UnaryOperator op);

class UnaryOperation : public NodeImpl<UnaryOperation, Expression,
                                       NodeKind::Expression_UnaryOperation> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  UnaryOperator getOperator() const;

  void setOperator(UnaryOperator newOp);

  Expression *getOperand() const;

  void setOperand(std::unique_ptr<Expression> node);

private:
  UnaryOperator op = UnaryOperator::Minus;
  Child<Expression> operand;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_UNARYOPERATION_H
