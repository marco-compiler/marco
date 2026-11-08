#ifndef MARCO_AST_MODELICA_NODE_H
#define MARCO_AST_MODELICA_NODE_H

#include "marco/AST/ASTNode.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace marco::ast::modelica {
/// Kinds of the nodes of the Modelica AST. Ranges of kinds are used to
/// implement the LLVM-style RTTI of the abstract nodes.
enum class NodeKind : int32_t {
  StoredDefinition,

  // Classes and their elements.
  ClassDefinition,
  ComponentClause,
  ComponentDeclaration,
  ExtendsClause,
  ImportClause,
  EquationSection,
  AlgorithmSection,
  ExternalClause,
  EnumerationLiteral,
  ConstrainingClause,

  // Modifications.
  Modification,
  ClassModification,
  Argument,
  Argument_ElementModification,
  Argument_ElementRedeclaration,
  Argument_InheritanceModification,
  Argument_Last,

  // Equations.
  Equation,
  Equation_Equality,
  Equation_Call,
  Equation_Connect,
  Equation_If,
  Equation_For,
  Equation_When,
  Equation_Last,

  // Statements.
  Statement,
  Statement_Assignment,
  Statement_MultipleAssignment,
  Statement_Call,
  Statement_Break,
  Statement_Return,
  Statement_If,
  Statement_For,
  Statement_While,
  Statement_When,
  Statement_Last,

  // Expressions.
  Expression,
  Expression_IntegerLiteral,
  Expression_RealLiteral,
  Expression_StringLiteral,
  Expression_BooleanLiteral,
  Expression_Time,
  Expression_End,
  Expression_Colon,
  Expression_ComponentReference,
  Expression_UnaryOperation,
  Expression_BinaryOperation,
  Expression_Range,
  Expression_If,
  Expression_Call,
  Expression_ArrayConstructor,
  Expression_ArrayConcatenation,
  Expression_OutputList,
  Expression_Subscripted,
  Expression_MemberAccess,
  Expression_FunctionPartialApplication,
  Expression_Last,

  // Auxiliary nodes.
  ForIndex,
  NamedArgument
};

llvm::StringRef toString(NodeKind kind);

class Node;

/// Owning pointer to a child node, which is deep-copied when copied.
template <typename T>
class Child {
public:
  Child() = default;

  Child(std::unique_ptr<T> node) : node(std::move(node)) {}

  Child(const Child &other)
      : node(other.node ? std::unique_ptr<T>(static_cast<T *>(
                              other.node->clone().release()))
                        : nullptr) {}

  Child(Child &&other) = default;

  Child &operator=(const Child &other) {
    if (this != &other) {
      Child copy(other);
      node = std::move(copy.node);
    }

    return *this;
  }

  Child &operator=(Child &&other) = default;

  Child &operator=(std::unique_ptr<T> other) {
    node = std::move(other);
    return *this;
  }

  explicit operator bool() const { return static_cast<bool>(node); }

  T *get() const { return node.get(); }

  T *operator->() const { return node.get(); }

  T &operator*() const { return *node; }

private:
  std::unique_ptr<T> node;
};

/// Owning list of child nodes.
template <typename T>
using Children = std::vector<Child<T>>;

/// Dotted name, as in type specifiers and modifications
/// (e.g., `.Modelica.Units.SI.Voltage`).
struct Name {
  /// Whether the name starts with a dot, meaning a global name.
  bool global = false;

  llvm::SmallVector<std::string, 3> parts;

  SourceRange location = SourceRange::unknown();

  std::string toString() const;
};

/// Base class of the nodes of the Modelica AST.
class Node : public ASTNode {
public:
  Node(NodeKind kind, SourceRange location) : ASTNode(kind, location) {}

  NodeKind getNodeKind() const { return getKind<NodeKind>(); }

  /// Invoke the callback on each direct child of the node.
  virtual void walkChildren(llvm::function_ref<void(Node *)> callback) = 0;

  /// Set the parent of the direct children of the node, and recursively of
  /// their descendants.
  void setParentOfChildren();

  llvm::json::Value toJSON() const override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;
};

/// Helper class implementing the methods common to all the concrete nodes.
template <typename Derived, typename Base, NodeKind Kind>
class NodeImpl : public Base {
public:
  explicit NodeImpl(SourceRange location) : Base(Kind, location) {}

  static bool classof(const ASTNode *node) {
    return node->getKind<NodeKind>() == Kind;
  }

  std::unique_ptr<ASTNode> clone() const override {
    auto result =
        std::make_unique<Derived>(static_cast<const Derived &>(*this));

    result->setParent(nullptr);
    result->setParentOfChildren();
    return result;
  }
};

/// Invoke a callback on a child, if present.
template <typename T>
void walk(const Child<T> &child, llvm::function_ref<void(Node *)> callback) {
  if (child) {
    callback(child.get());
  }
}

/// Invoke a callback on each child of a list.
template <typename T>
void walk(const Children<T> &children,
          llvm::function_ref<void(Node *)> callback) {
  for (const Child<T> &child : children) {
    walk(child, callback);
  }
}

/// JSON conversion utilities.
llvm::json::Value toJSON(const Name &name);
llvm::json::Value toJSON(const Node *node);

template <typename T>
llvm::json::Value toJSON(const Children<T> &children) {
  llvm::json::Array array;

  for (const Child<T> &child : children) {
    array.push_back(toJSON(child.get()));
  }

  return array;
}
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_NODE_H
