#ifndef MARCO_AST_BASEMODELICA_NODE_CLASS_H
#define MARCO_AST_BASEMODELICA_NODE_CLASS_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"

namespace marco::ast::bmodelica {
class Annotation;
class ExternalFunctionCall;

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

  bool isExternal() const;

  void setExternal(bool isExternal);

  llvm::StringRef getExternalLanguage() const;

  void setExternalLanguage(llvm::StringRef language);

  bool hasExternalFunctionCall() const;

  ExternalFunctionCall *getExternalFunctionCall();

  const ExternalFunctionCall *getExternalFunctionCall() const;

  void setExternalFunctionCall(std::unique_ptr<ASTNode> node);

  bool hasExternalAnnotation() const;

  Annotation *getExternalAnnotation();

  const Annotation *getExternalAnnotation() const;

  void setExternalAnnotation(std::unique_ptr<ASTNode> node);

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
  bool external{false};
  std::string externalLanguage{"C"};
  std::unique_ptr<ASTNode> externalFunctionCall;
  std::unique_ptr<ASTNode> externalAnnotation;
  std::unique_ptr<ASTNode> annotation;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_CLASS_H
