#ifndef MARCO_AST_BASEMODELICA_ARRAYCONSTANT_H
#define MARCO_AST_BASEMODELICA_ARRAYCONSTANT_H

#include "marco/AST/BaseModelica/ArrayGenerator.h"
#include "marco/AST/BaseModelica/Type.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast::bmodelica {
class ArrayGenerator;

class ArrayConstant : public ArrayGenerator {
public:
  explicit ArrayConstant(SourceRange location);

  ArrayConstant(const ArrayConstant &other);

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() ==
           ASTNodeKind::Expression_ArrayGenerator_ArrayConstant;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  size_t size() const;

  Expression *operator[](size_t index);

  const Expression *operator[](size_t index) const;

  void setValues(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>> values;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_ARRAYCONSTANT_H
