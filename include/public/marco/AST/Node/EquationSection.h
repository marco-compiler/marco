#ifndef PUBLIC_MARCO_AST_NODE_EQUATIONSECTION_H
#define PUBLIC_MARCO_AST_NODE_EQUATIONSECTION_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/Parser/Location.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <cstddef>
#include <memory>

namespace marco::ast {
class Equation;

class EquationSection : public ASTNode {
public:
  explicit EquationSection(SourceRange location);

  EquationSection(const EquationSection &other);

  ~EquationSection() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::EquationSection;
  }

  std::unique_ptr<ASTNode> clone() const override;

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
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_EQUATIONSECTION_H
