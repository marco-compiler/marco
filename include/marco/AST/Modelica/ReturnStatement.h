#ifndef MARCO_AST_MODELICA_RETURNSTATEMENT_H
#define MARCO_AST_MODELICA_RETURNSTATEMENT_H

#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
class ReturnStatement
    : public NodeImpl<ReturnStatement, Statement, NodeKind::Statement_Return> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_RETURNSTATEMENT_H
