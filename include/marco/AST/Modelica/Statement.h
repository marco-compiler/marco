#ifndef MARCO_AST_MODELICA_STATEMENT_H
#define MARCO_AST_MODELICA_STATEMENT_H

#include "marco/AST/Modelica/Description.h"
#include "marco/AST/Modelica/Node.h"

namespace marco::ast::modelica {
/// Base class of the statements.
class Statement : public Node {
public:
  using Node::Node;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  static bool classof(const ASTNode *node);

  Description &getDescription();

  const Description &getDescription() const;

protected:
  Description description;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_STATEMENT_H
