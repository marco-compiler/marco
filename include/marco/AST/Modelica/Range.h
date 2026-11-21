#ifndef MARCO_AST_MODELICA_RANGE_H
#define MARCO_AST_MODELICA_RANGE_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Range expression `start:stop` or `start:step:stop`.
class Range : public NodeImpl<Range, Expression, NodeKind::Expression_Range> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  Expression *getStart() const;

  void setStart(std::unique_ptr<Expression> node);

  /// Get the step, if specified.
  Expression *getStep() const;

  void setStep(std::unique_ptr<Expression> node);

  Expression *getStop() const;

  void setStop(std::unique_ptr<Expression> node);

private:
  Child<Expression> start;
  Child<Expression> step;
  Child<Expression> stop;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_RANGE_H
