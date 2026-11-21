#ifndef MARCO_AST_MODELICA_COLON_H
#define MARCO_AST_MODELICA_COLON_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// The `:` subscript, selecting a whole dimension.
class Colon : public NodeImpl<Colon, Expression, NodeKind::Expression_Colon> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override;

};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_COLON_H
