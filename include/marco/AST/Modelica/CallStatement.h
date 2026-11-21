#ifndef MARCO_AST_MODELICA_CALLSTATEMENT_H
#define MARCO_AST_MODELICA_CALLSTATEMENT_H

#include "marco/AST/Modelica/Call.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
class CallStatement
    : public NodeImpl<CallStatement, Statement, NodeKind::Statement_Call> {
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

#endif // MARCO_AST_MODELICA_CALLSTATEMENT_H
