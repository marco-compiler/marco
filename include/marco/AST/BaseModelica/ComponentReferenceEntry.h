#ifndef MARCO_AST_BASEMODELICA_COMPONENTREFERENCEENTRY_H
#define MARCO_AST_BASEMODELICA_COMPONENTREFERENCEENTRY_H

#include "marco/AST/BaseModelica/ASTNode.h"
#include "marco/AST/BaseModelica/Type.h"
#include <string>

namespace marco::ast::bmodelica {
class Subscript;

class ComponentReferenceEntry : public ASTNode {
public:
  explicit ComponentReferenceEntry(SourceRange location);

  ComponentReferenceEntry(const ComponentReferenceEntry &other);

  ~ComponentReferenceEntry() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::ComponentReferenceEntry;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  llvm::StringRef getName() const;

  void setName(llvm::StringRef newName);

  size_t getNumOfSubscripts() const;

  Subscript *getSubscript(size_t index);

  const Subscript *getSubscript(size_t index) const;

  void setSubscripts(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  std::string name;
  llvm::SmallVector<std::unique_ptr<ASTNode>> subscripts;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_COMPONENTREFERENCEENTRY_H
