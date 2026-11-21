#ifndef MARCO_AST_MODELICA_TIME_H
#define MARCO_AST_MODELICA_TIME_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// The built-in variable `time`.
class Time : public NodeImpl<Time, Expression, NodeKind::Expression_Time> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override;

};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_TIME_H
