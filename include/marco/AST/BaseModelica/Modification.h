#ifndef MARCO_AST_BASEMODELICA_MODIFICATION_H
#define MARCO_AST_BASEMODELICA_MODIFICATION_H

#include "marco/AST/BaseModelica/ASTNode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <variant>

namespace marco::ast::bmodelica {
class Argument;
class ClassModification;
class ElementModification;
class ElementRedeclaration;
class ElementReplaceable;
class Expression;
class Modification;
class ArrayConstant;

class Modification : public ASTNode {
public:
  explicit Modification(SourceRange location);

  Modification(const Modification &other);

  ~Modification() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::Modification;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool hasClassModification() const;

  ClassModification *getClassModification();

  const ClassModification *getClassModification() const;

  void setClassModification(std::unique_ptr<ASTNode> node);

  bool hasExpression() const;

  Expression *getExpression();

  const Expression *getExpression() const;

  void setExpression(std::unique_ptr<ASTNode> node);

  /// @name Forwarded methods
  /// {

  bool hasStartExpression() const;

  Expression *getStartExpression();

  const Expression *getStartExpression() const;

  // FIXME: might not be uniform in case of arrays
  std::optional<bool> getFixedProperty() const;

  bool getEachProperty() const;

  /// }

private:
  std::unique_ptr<ASTNode> classModification;
  std::unique_ptr<ASTNode> expression;
};

class ClassModification : public ASTNode {
public:
  explicit ClassModification(SourceRange location);

  ClassModification(const ClassModification &other);

  ~ClassModification() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::ClassModification;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getArguments() const;

  void setArguments(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

  /// @name Predefined properties
  /// {

  bool hasStartExpression() const;

  Expression *getStartExpression();

  const Expression *getStartExpression() const;

  // FIXME: might not be uniform in case of arrays
  std::optional<bool> getFixedProperty() const;

  bool getEachProperty() const;

  /// }

private:
  llvm::SmallVector<std::unique_ptr<ASTNode>> arguments;

  static std::optional<bool>
  isArrayUniformConstBool(const ArrayConstant *array);
};

class Argument : public ASTNode {
public:
  using ASTNode::ASTNode;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() >= ASTNodeKind::Argument &&
           node->getKind<ASTNodeKind>() <= ASTNodeKind::Argument_LastArgument;
  }
};

class ElementModification : public Argument {
public:
  explicit ElementModification(SourceRange location);

  ElementModification(const ElementModification &other);

  ~ElementModification() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() ==
           ASTNodeKind::Argument_ElementModification;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool hasEachProperty() const;

  void setEachProperty(bool value);

  // FIXME: might not be uniform in case of arrays
  bool hasFinalProperty() const;

  // FIXME: might not be uniform in case of arrays
  void setFinalProperty(bool value);

  llvm::StringRef getName() const;

  void setName(llvm::StringRef newName);

  bool hasModification() const;

  Modification *getModification();

  const Modification *getModification() const;

  void setModification(std::unique_ptr<ASTNode> node);

private:
  bool each{false};
  bool final{false};
  std::string name;
  std::unique_ptr<ASTNode> modification;
};

// TODO: ElementReplaceable
class ElementReplaceable : public Argument {
public:
  explicit ElementReplaceable(SourceRange location);

  ElementReplaceable(const ElementReplaceable &other);

  ~ElementReplaceable() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() ==
           ASTNodeKind::Argument_ElementReplaceable;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};

// TODO: ElementRedeclaration
class ElementRedeclaration : public Argument {
public:
  explicit ElementRedeclaration(SourceRange location);

  ElementRedeclaration(const ElementRedeclaration &other);

  ~ElementRedeclaration() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() ==
           ASTNodeKind::Argument_ElementRedeclaration;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_MODIFICATION_H
