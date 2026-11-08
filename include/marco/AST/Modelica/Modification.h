#ifndef MARCO_AST_MODELICA_MODIFICATION_H
#define MARCO_AST_MODELICA_MODIFICATION_H

#include "marco/AST/Modelica/Expression.h"

namespace marco::ast::modelica {
class ClassModification;

/// Modification: an optional class modification, followed by
/// an optional value (`= expression` or `= break`).
class Modification
    : public NodeImpl<Modification, Node, NodeKind::Modification> {
public:
  using NodeImpl::NodeImpl;

  ClassModification *getClassModification() const {
    return classModification.get();
  }

  void setClassModification(std::unique_ptr<ClassModification> node);

  /// Get the value, if any.
  Expression *getValue() const { return value.get(); }
  void setValue(std::unique_ptr<Expression> node) { value = std::move(node); }

  /// Whether the value is `break`.
  bool isBreak() const { return breakValue; }
  void setBreak(bool isBreak) { breakValue = isBreak; }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<ClassModification> classModification;
  Child<Expression> value;
  bool breakValue = false;
};

/// Base class of the arguments of a class modification.
class Argument : public Node {
public:
  using Node::Node;

  static bool classof(const ASTNode *node) {
    return node->getKind<NodeKind>() >= NodeKind::Argument &&
           node->getKind<NodeKind>() <= NodeKind::Argument_Last;
  }
};

/// Class modification: a parenthesized list of arguments. It is also used to
/// represent annotations.
class ClassModification
    : public NodeImpl<ClassModification, Node, NodeKind::ClassModification> {
public:
  using NodeImpl::NodeImpl;

  const Children<Argument> &getArguments() const { return arguments; }
  void addArgument(std::unique_ptr<Argument> node) {
    arguments.emplace_back(std::move(node));
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Argument> arguments;
};

/// Modification of an element (`each final a.b(c = 1) = 2 "description"`).
class ElementModification
    : public NodeImpl<ElementModification, Argument,
                      NodeKind::Argument_ElementModification> {
public:
  using NodeImpl::NodeImpl;

  bool isEach() const { return each; }
  void setEach(bool value) { each = value; }

  bool isFinal() const { return final; }
  void setFinal(bool value) { final = value; }

  const Name &getName() const { return name; }
  void setName(Name newName) { name = std::move(newName); }

  Modification *getModification() const { return modification.get(); }
  void setModification(std::unique_ptr<Modification> node) {
    modification = std::move(node);
  }

  llvm::StringRef getDescription() const { return description; }
  void setDescription(std::string text) { description = std::move(text); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  bool each = false;
  bool final = false;
  Name name;
  Child<Modification> modification;
  std::string description;
};

/// Constraining clause (`constrainedby Type(modification) "description"`).
class ConstrainingClause
    : public NodeImpl<ConstrainingClause, Node, NodeKind::ConstrainingClause> {
public:
  using NodeImpl::NodeImpl;

  const Name &getTypeSpecifier() const { return typeSpecifier; }
  void setTypeSpecifier(Name name) { typeSpecifier = std::move(name); }

  ClassModification *getClassModification() const {
    return classModification.get();
  }

  void setClassModification(std::unique_ptr<ClassModification> node) {
    classModification = std::move(node);
  }

  llvm::StringRef getDescription() const { return description; }
  void setDescription(std::string text) { description = std::move(text); }

  ClassModification *getAnnotation() const { return annotation.get(); }
  void setAnnotation(std::unique_ptr<ClassModification> node) {
    annotation = std::move(node);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Name typeSpecifier;
  Child<ClassModification> classModification;
  std::string description;
  Child<ClassModification> annotation;
};

/// Declaration of an element within a modification, that is an
/// element-redeclaration or an element-replaceable. The
/// element is either a short class definition (ClassDefinition) or a
/// component clause with a single declaration (ComponentClause).
class ElementRedeclaration
    : public NodeImpl<ElementRedeclaration, Argument,
                      NodeKind::Argument_ElementRedeclaration> {
public:
  using NodeImpl::NodeImpl;

  bool isRedeclare() const { return redeclare; }
  void setRedeclare(bool value) { redeclare = value; }

  bool isEach() const { return each; }
  void setEach(bool value) { each = value; }

  bool isFinal() const { return final; }
  void setFinal(bool value) { final = value; }

  bool isReplaceable() const { return replaceable; }
  void setReplaceable(bool value) { replaceable = value; }

  Node *getElement() const { return element.get(); }
  void setElement(std::unique_ptr<Node> node) { element = std::move(node); }

  ConstrainingClause *getConstrainingClause() const {
    return constrainingClause.get();
  }

  void setConstrainingClause(std::unique_ptr<ConstrainingClause> node) {
    constrainingClause = std::move(node);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  bool redeclare = false;
  bool each = false;
  bool final = false;
  bool replaceable = false;
  Child<Node> element;
  Child<ConstrainingClause> constrainingClause;
};

/// Deselection of selective model extension (`break name` or
/// `break connect(a, b)`).
class InheritanceModification
    : public NodeImpl<InheritanceModification, Argument,
                      NodeKind::Argument_InheritanceModification> {
public:
  using NodeImpl::NodeImpl;

  /// Name of the deselected component, empty for connection deselections.
  llvm::StringRef getComponent() const { return component; }
  void setComponent(std::string name) { component = std::move(name); }

  /// Connected elements of a connection deselection.
  ComponentReference *getConnectLhs() const { return connectLhs.get(); }
  ComponentReference *getConnectRhs() const { return connectRhs.get(); }

  void setConnect(std::unique_ptr<ComponentReference> lhs,
                  std::unique_ptr<ComponentReference> rhs) {
    connectLhs = std::move(lhs);
    connectRhs = std::move(rhs);
  }

  bool isConnect() const { return static_cast<bool>(connectLhs); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::string component;
  Child<ComponentReference> connectLhs;
  Child<ComponentReference> connectRhs;
};

/// Description of a declaration, equation or statement: a description string
/// and an optional annotation.
struct Description {
  std::string text;
  Child<ClassModification> annotation;

  void walkChildren(llvm::function_ref<void(Node *)> callback) {
    walk(annotation, callback);
  }

  void addToJSON(llvm::json::Object &obj) const;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_MODIFICATION_H
