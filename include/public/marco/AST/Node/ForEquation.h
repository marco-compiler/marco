#ifndef PUBLIC_MARCO_AST_NODE_FOREQUATION_H
#define PUBLIC_MARCO_AST_NODE_FOREQUATION_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Equation.h"
#include "marco/Parser/Location.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <cstddef>
#include <memory>

namespace marco::ast {
class Equation;
class ForIndex;

class ForEquation : public Equation {
public:
  explicit ForEquation(SourceRange location);

  ForEquation(const ForEquation &other);

  ~ForEquation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Equation_For;
  }

  std::unique_ptr<ASTNode> clone() const override;

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
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_FOREQUATION_H
