#ifndef MARCO_AST_NODE_CLASS_H
#define MARCO_AST_NODE_CLASS_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast {
class Annotation;

class Class : public ASTNode {
public:
  using ASTNode::ASTNode;

  Class(const Class &other);

  ~Class() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() >= ASTNode::Kind::Class &&
           node->getKind() <= ASTNode::Kind::Class_LastClass;
  }

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  /// Get the name.
  llvm::StringRef getName() const;

  /// Set the name.
  void setName(llvm::StringRef newName);

  /// Get the variables.
  llvm::ArrayRef<std::unique_ptr<ASTNode>> getVariables() const;

  /// Set the variables.
  void setVariables(llvm::ArrayRef<std::unique_ptr<ASTNode>> newVariables);

  /// Get the equation sections.
  llvm::ArrayRef<std::unique_ptr<ASTNode>> getEquationSections() const;

  /// Set the equation sections.
  void setEquationSections(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> newEquationSections);

  /// Get the 'algorithm' blocks.
  llvm::ArrayRef<std::unique_ptr<ASTNode>> getAlgorithms() const;

  /// Set the 'algorithm' blocks.
  void setAlgorithms(llvm::ArrayRef<std::unique_ptr<ASTNode>> newAlgorithms);

  /// Get the inner classes.
  llvm::ArrayRef<std::unique_ptr<ASTNode>> getInnerClasses() const;

  /// Set the inner classes.
  void
  setInnerClasses(llvm::ArrayRef<std::unique_ptr<ASTNode>> newInnerClasses);

  bool hasAnnotation() const;

  Annotation *getAnnotation();

  const Annotation *getAnnotation() const;

  void setAnnotation(std::unique_ptr<ASTNode> node);

private:
  std::string name;
  llvm::SmallVector<std::unique_ptr<ASTNode>> variables;
  llvm::SmallVector<std::unique_ptr<ASTNode>> equationSections;
  llvm::SmallVector<std::unique_ptr<ASTNode>> algorithms;
  llvm::SmallVector<std::unique_ptr<ASTNode>> innerClasses;
  std::unique_ptr<ASTNode> annotation;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_CLASS_H
