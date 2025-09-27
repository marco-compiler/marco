#ifndef MARCO_AST_BASEMODELICA_TYPE_H
#define MARCO_AST_BASEMODELICA_TYPE_H

#include "marco/AST/BaseModelica/ASTNode.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

namespace marco::ast::bmodelica {
class ArrayDimension;

class VariableType : public ASTNode {
public:
  using ASTNode::ASTNode;

  VariableType(const VariableType &other);

  ~VariableType() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() >= ASTNodeKind::VariableType &&
           node->getKind<ASTNodeKind>() <=
               ASTNodeKind::VariableType_LastVariableType;
  }

  void addJSONProperties(llvm::json::Object &obj) const override;

  size_t getRank() const;

  ArrayDimension *operator[](size_t index);

  const ArrayDimension *operator[](size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getDimensions() const;

  void setDimensions(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  bool hasConstantShape() const;

  bool isScalar() const;

  std::unique_ptr<ASTNode> subscript(size_t times) const;

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> dimensions;
};

class BuiltInType : public VariableType {
public:
  enum class Kind { Boolean, Integer, Real, String };

  BuiltInType(SourceRange location);

  BuiltInType(const BuiltInType &other);

  ~BuiltInType() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::VariableType_BuiltIn;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Kind getBuiltInTypeKind() const;

  void setBuiltInTypeKind(Kind newKind);

  bool isNumeric() const;

private:
  Kind kind;
};

class UserDefinedType : public VariableType {
public:
  UserDefinedType(SourceRange location);

  UserDefinedType(const UserDefinedType &other);

  ~UserDefinedType() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() ==
           ASTNodeKind::VariableType_UserDefined;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool isGlobalLookup() const;

  void setGlobalLookup(bool global);

  size_t getPathLength() const;

  llvm::StringRef getElement(size_t index) const;

  void setPath(llvm::ArrayRef<std::string> newPath);

private:
  bool globalLookup;
  llvm::SmallVector<std::string> path;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_TYPE_H
