#ifndef MARCO_AST_BASEMODELICA_FOREQUATION_H
#define MARCO_AST_BASEMODELICA_FOREQUATION_H

#include "marco/AST/BaseModelica/Equation.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast::bmodelica {
class Equation;
class ForIndex;

class ForEquation : public Equation {
public:
  explicit ForEquation(SourceRange location);

  ForEquation(const ForEquation &other);

  ~ForEquation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::Equation_For;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  size_t getNumOfForIndices() const;

  ForIndex *getForIndex(size_t index);

  const ForIndex *getForIndex(size_t index) const;

  void setForIndices(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  size_t getNumOfEquations() const;

  Equation *getEquation(size_t index);

  const Equation *getEquation(size_t index) const;

  void setEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> forIndices;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> equations;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_FOREQUATION_H
