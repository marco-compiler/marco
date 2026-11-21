#ifndef MARCO_AST_MODELICA_CALLEQUATION_H
#define MARCO_AST_MODELICA_CALLEQUATION_H

#include "marco/AST/Modelica/Call.h"
#include "marco/AST/Modelica/Equation.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Function call used as equation, such as `assert(...)` or
/// `Connections.branch(...)`.
class CallEquation
    : public NodeImpl<CallEquation, Equation, NodeKind::Equation_Call> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  Call *getCall() const;

  void setCall(std::unique_ptr<Call> node);

private:
  Child<Call> call;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_CALLEQUATION_H
