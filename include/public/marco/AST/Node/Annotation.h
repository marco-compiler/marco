#ifndef MARCO_AST_NODE_ANNOTATION_H
#define MARCO_AST_NODE_ANNOTATION_H

#include "marco/AST/Node/ASTNode.h"
#include <memory>

namespace marco::ast {
class ClassModification;
class DerivativeAnnotation;
class InverseFunctionAnnotation;

class Annotation : public ASTNode {
public:
  explicit Annotation(SourceRange location);

  Annotation(const Annotation &other);

  ~Annotation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Annotation;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  ClassModification *getProperties();

  const ClassModification *getProperties() const;

  void setProperties(std::unique_ptr<ASTNode> newProperties);

  bool getInlineProperty() const;

  bool hasDerivativeAnnotation() const;
  DerivativeAnnotation getDerivativeAnnotation() const;

  InverseFunctionAnnotation getInverseFunctionAnnotation() const;

private:
  std::unique_ptr<ASTNode> properties;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_ANNOTATION_H
