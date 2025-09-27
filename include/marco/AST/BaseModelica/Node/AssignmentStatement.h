#ifndef MARCO_AST_BASEMODELICA_NODE_ASSIGNMENTSTATEMENT_H
#define MARCO_AST_BASEMODELICA_NODE_ASSIGNMENTSTATEMENT_H

#include "marco/AST/BaseModelica/Node/Statement.h"

namespace marco::ast::bmodelica {
class Expression;
class Tuple;

class AssignmentStatement : public Statement {
public:
  explicit AssignmentStatement(SourceRange location);

  AssignmentStatement(const AssignmentStatement &other);

  ~AssignmentStatement() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Statement_Assignment;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Tuple *getDestinations();

  const Tuple *getDestinations() const;

  void setDestinations(std::unique_ptr<ASTNode> nodes);

  Expression *getExpression();

  const Expression *getExpression() const;

  void setExpression(std::unique_ptr<ASTNode> node);

private:
  // Where the result of the expression has to be stored.
  // It is always a tuple, because functions may have multiple outputs.
  std::unique_ptr<ASTNode> destinations;

  // Right-hand side expression of the assignment.
  std::unique_ptr<ASTNode> expression;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_ASSIGNMENTSTATEMENT_H
