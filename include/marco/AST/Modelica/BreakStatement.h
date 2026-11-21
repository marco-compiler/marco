#ifndef MARCO_AST_MODELICA_BREAKSTATEMENT_H
#define MARCO_AST_MODELICA_BREAKSTATEMENT_H

#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
class BreakStatement
    : public NodeImpl<BreakStatement, Statement, NodeKind::Statement_Break> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_BREAKSTATEMENT_H
