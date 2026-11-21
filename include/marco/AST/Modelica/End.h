#ifndef MARCO_AST_MODELICA_END_H
#define MARCO_AST_MODELICA_END_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// The `end` keyword within array subscripts.
class End : public NodeImpl<End, Expression, NodeKind::Expression_End> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override;

};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_END_H
