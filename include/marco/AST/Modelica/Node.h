#ifndef MARCO_AST_MODELICA_NODE_H
#define MARCO_AST_MODELICA_NODE_H

#include "marco/AST/ASTNode.h"
#include "marco/AST/Modelica/Child.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace marco::ast::modelica {
/// Kinds of the nodes of the Modelica AST.
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

/// Base class of the nodes of the Modelica AST.
class Node : public ASTNode {
public:
  Node(NodeKind kind, SourceRange location);

  NodeKind getNodeKind() const;

  /// Invoke the callback on each direct child of the node.
  virtual void walkChildren(llvm::function_ref<void(Node *)> callback) = 0;

  /// Set the parent of the direct children of the node, and recursively of
  /// their descendants.
  void setParentOfChildren();

  llvm::json::Value toJSON() const override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;
};

/// Invoke a callback on a child, if present.
template <typename T>
void walk(const Child<T> &child, llvm::function_ref<void(Node *)> callback) {
  if (child) {
    callback(child.get());
  }
}

/// Invoke a callback on each child of a list, whatever its container.
template <typename Range>
auto walk(const Range &children, llvm::function_ref<void(Node *)> callback)
    -> decltype(std::begin(children), void()) {
  for (const auto &child : children) {
    walk(child, callback);
  }
}

/// JSON conversion utilities.
llvm::json::Value toJSON(const Node *node);

template <typename T>
llvm::json::Value toJSON(const Child<T> &child) {
  return toJSON(child.get());
}

/// Get the JSON array of a list of children, or of a list of lists of
/// children, whatever their containers.
template <typename Range>
auto toJSON(const Range &children)
    -> decltype(std::begin(children), llvm::json::Value(nullptr)) {
  llvm::json::Array array;

  for (const auto &child : children) {
    array.push_back(toJSON(child));
  }

  return array;
}
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_NODE_H
