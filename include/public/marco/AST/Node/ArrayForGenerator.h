#ifndef PUBLIC_MARCO_AST_NODE_ARRAYFORGENERATOR_H
#define PUBLIC_MARCO_AST_NODE_ARRAYFORGENERATOR_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/ArrayGenerator.h"
#include "marco/Parser/Location.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <memory>

namespace marco::ast {
class ArrayGenerator;
class ForIndex;

class ArrayForGenerator : public ArrayGenerator {
public:
  explicit ArrayForGenerator(SourceRange location);

  ArrayForGenerator(const ArrayForGenerator &other);

  static bool classof(const ASTNode *node) {
    return node->getKind() ==
           ASTNode::Kind::Expression_ArrayGenerator_ArrayForGenerator;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getValue();

  const Expression *getValue() const;

  void setValue(std::unique_ptr<ASTNode> node);

  unsigned getNumIndices() const;

  ForIndex *getIndex(unsigned index);

  const ForIndex *getIndex(unsigned index) const;

  void setIndices(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  std::unique_ptr<ASTNode> value;
  llvm::SmallVector<std::unique_ptr<ASTNode>> indices;
};
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_ARRAYFORGENERATOR_H
