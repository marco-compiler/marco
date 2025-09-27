#ifndef MARCO_AST_BASEMODELICA_NODE_ARRAY_FOR_GENERATOR_H
#define MARCO_AST_BASEMODELICA_NODE_ARRAY_FOR_GENERATOR_H

#include "marco/AST/BaseModelica/Node/ArrayGenerator.h"
#include "marco/AST/BaseModelica/Node/Type.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast::bmodelica {
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
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_ARRAY_FOR_GENERATOR_H
