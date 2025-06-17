#ifndef MARCO_AST_NODE_IFEQUATION_H
#define MARCO_AST_NODE_IFEQUATION_H

#include "marco/AST/Node/Equation.h"

namespace marco::ast {
class Expression;

class IfEquation : public Equation {
public:
  IfEquation(SourceRange location);

  IfEquation(const IfEquation &other);

  ~IfEquation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Equation_If;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getIfCondition();

  const Expression *getIfCondition() const;

  void setIfCondition(std::unique_ptr<ASTNode> node);

  size_t getNumOfIfEquations() const;

  Equation *getIfEquation(size_t index);

  const Equation *getIfEquation(size_t index) const;

  void setIfEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  size_t getNumOfElseIfConditions() const;

  Expression *getElseIfCondition(size_t index);

  const Expression *getElseIfCondition(size_t index) const;

  void setElseIfConditions(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  size_t getNumOfElseIfEquations(size_t condition) const;

  Equation *getElseIfEquation(size_t condition, size_t equation);

  const Equation *getElseIfEquation(size_t condition, size_t equation) const;

  void setElseIfEquations(size_t condition,
                          llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  size_t getNumOfElseEquations() const;

  bool hasElseEquations() const;

  Equation *getElseEquation(size_t index);

  const Equation *getElseEquation(size_t index) const;

  void setElseEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  std::unique_ptr<ASTNode> ifCondition;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> ifEquations;

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseIfConditions;
  llvm::SmallVector<llvm::SmallVector<std::unique_ptr<ASTNode>, 3>>
      elseIfEquations;

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseEquations;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_IFEQUATION_H
