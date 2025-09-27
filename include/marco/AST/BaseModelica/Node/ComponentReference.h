#ifndef MARCO_AST_BASEMODELICA_NODE_COMPONENTREFERENCE_H
#define MARCO_AST_BASEMODELICA_NODE_COMPONENTREFERENCE_H

#include "marco/AST/BaseModelica/Node/Expression.h"
#include "marco/AST/BaseModelica/Node/Type.h"
#include <string>

namespace marco::ast::bmodelica {
class ComponentReferenceEntry;

class ComponentReference : public Expression {
public:
  explicit ComponentReference(SourceRange location);

  ComponentReference(const ComponentReference &other);

  ~ComponentReference() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Expression_ComponentReference;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool isLValue() const override;

  bool isDummy() const;

  void setDummy(bool value);

  bool isGlobalLookup() const;

  void setGlobalLookup(bool global);

  size_t getPathLength() const;

  ComponentReferenceEntry *getElement(size_t index);

  const ComponentReferenceEntry *getElement(size_t index) const;

  void setPath(llvm::ArrayRef<std::unique_ptr<ASTNode>> newPath);

  std::string getName() const;

private:
  bool dummy{false};
  bool globalLookup{false};
  llvm::SmallVector<std::unique_ptr<ASTNode>> path;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_COMPONENTREFERENCE_H
