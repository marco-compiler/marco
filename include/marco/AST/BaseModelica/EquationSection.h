#ifndef MARCO_AST_BASEMODELICA_EQUATIONSECTION_H
#define MARCO_AST_BASEMODELICA_EQUATIONSECTION_H

#include "marco/AST/BaseModelica/ASTNode.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast::bmodelica {
class Equation;

class EquationSection : public ASTNode {
public:
  explicit EquationSection(SourceRange location);

  EquationSection(const EquationSection &other);

  ~EquationSection() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::EquationSection;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool isInitial() const;

  void setInitial(bool value);

  size_t getNumOfEquations() const;

  bool empty() const;

  Equation *getEquation(size_t index);

  const Equation *getEquation(size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getEquations();

  void setEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  bool initial{false};
  llvm::SmallVector<std::unique_ptr<ASTNode>> equations;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_EQUATIONSECTION_H
