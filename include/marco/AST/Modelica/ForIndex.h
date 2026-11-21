#ifndef MARCO_AST_MODELICA_FORINDEX_H
#define MARCO_AST_MODELICA_FORINDEX_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Iterator of for-loops, reductions and array constructors
/// (`i in 1:n`). The range is null if implicit.
class ForIndex : public NodeImpl<ForIndex, Node, NodeKind::ForIndex> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  llvm::StringRef getName() const;

  void setName(std::string newName);

  Expression *getRange() const;

  void setRange(std::unique_ptr<Expression> node);

private:
  std::string name;
  Child<Expression> range;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_FORINDEX_H
