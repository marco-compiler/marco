#ifndef MARCO_AST_NODE_MODIFICATION_H
#define MARCO_AST_NODE_MODIFICATION_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <variant>

namespace marco::ast {
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
    return node->getKind() == ASTNode::Kind::Modification;
  }

  std::unique_ptr<ASTNode> clone() const override;

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
    return node->getKind() == ASTNode::Kind::ClassModification;
  }

  std::unique_ptr<ASTNode> clone() const override;

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
    return node->getKind() >= ASTNode::Kind::Argument &&
           node->getKind() <= ASTNode::Kind::Argument_LastArgument;
  }
};

class ElementModification : public Argument {
public:
  explicit ElementModification(SourceRange location);

  ElementModification(const ElementModification &other);

  ~ElementModification() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Argument_ElementModification;
  }

  std::unique_ptr<ASTNode> clone() const override;

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
  bool each;
  bool final;
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
    return node->getKind() == ASTNode::Kind::Argument_ElementReplaceable;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};

// TODO: ElementRedeclaration
class ElementRedeclaration : public Argument {
public:
  explicit ElementRedeclaration(SourceRange location);

  ElementRedeclaration(const ElementRedeclaration &other);

  ~ElementRedeclaration() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Argument_ElementRedeclaration;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_MODIFICATION_H
