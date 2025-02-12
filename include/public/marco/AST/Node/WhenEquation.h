#ifndef MARCO_AST_NODE_WHENEQUATION_H
#define MARCO_AST_NODE_WHENEQUATION_H

#include "marco/AST/Node/Equation.h"

namespace marco::ast {
class Expression;

class WhenEquation : public Equation {
public:
  WhenEquation(SourceRange location);

  WhenEquation(const WhenEquation &other);

  ~WhenEquation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Equation_When;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getWhenCondition();

  const Expression *getWhenCondition() const;

  void setWhenCondition(std::unique_ptr<ASTNode> node);

  size_t getNumOfWhenEquations() const;

  Equation *getWhenEquation(size_t index);

  const Equation *getWhenEquation(size_t index) const;

  void setWhenEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  size_t getNumOfElseWhenConditions() const;

  Expression *getElseWhenCondition(size_t index);

  const Expression *getElseWhenCondition(size_t index) const;

  void setElseWhenConditions(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  size_t getNumOfElseWhenEquations(size_t condition) const;

  Equation *getElseWhenEquation(size_t condition, size_t equation);

  const Equation *getElseWhenEquation(size_t condition, size_t equation) const;

  void setElseWhenEquations(size_t condition,
                            llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  size_t getNumOfElseEquations() const;

  bool hasElseEquations() const;

  Equation *getElseEquation(size_t index);

  const Equation *getElseEquation(size_t index) const;

  void setElseEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  std::unique_ptr<ASTNode> whenCondition;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> whenEquations;

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseWhenConditions;
  llvm::SmallVector<llvm::SmallVector<std::unique_ptr<ASTNode>, 3>>
      elseWhenEquations;

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseEquations;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_WHENEQUATION_H
