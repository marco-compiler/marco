#ifndef MARCO_AST_NODE_ASTNODE_H
#define MARCO_AST_NODE_ASTNODE_H

#include "marco/Parser/Location.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace marco::ast {
class ASTNode {
public:
  enum class Kind {
    Root,
    Algorithm,
    Annotation,
    Argument,
    Argument_ElementModification,
    Argument_ElementRedeclaration,
    Argument_ElementReplaceable,
    Argument_LastArgument,
    ArrayDimension,
    Class,
    Class_Function,
    Class_Function_PartialDerFunction,
    Class_Function_StandardFunction,
    Class_Function_LastFunction,
    Class_Model,
    Class_Package,
    Class_Record,
    Class_LastClass,
    ClassModification,
    ComponentReferenceEntry,
    Equation,
    Equation_Equality,
    Equation_For,
    Equation_If,
    Equation_When,
    Equation_LastEquation,
    EquationSection,
    Expression,
    Expression_ArrayGenerator,
    Expression_ArrayGenerator_ArrayConstant,
    Expression_ArrayGenerator_ArrayForGenerator,
    Expression_ArrayGenerator_LastArrayGenerator,
    Expression_Call,
    Expression_ComponentReference,
    Expression_Constant,
    Expression_Operation,
    Expression_Reduction,
    Expression_Subscript,
    Expression_Tuple,
    Expression_LastExpression,
    ForIndex,
    FunctionArgument,
    FunctionArgument_Expression,
    FunctionArgument_Named,
    FunctionArgument_Reduction,
    FunctionArgument_LastFunctionArgument,
    Member,
    Modification,
    Statement,
    Statement_Assignment,
    Statement_Break,
    Statement_Call,
    Statement_For,
    Statement_If,
    Statement_Return,
    Statement_When,
    Statement_While,
    Statement_LastStatement,
    StatementsBlock,
    TypePrefix,
    VariableType,
    VariableType_BuiltIn,
    VariableType_UserDefined,
    VariableType_LastVariableType
  };

  ASTNode(Kind kind, SourceRange location, ASTNode *parent = nullptr);

  ASTNode(const ASTNode &other);

  virtual ~ASTNode() = 0;

  /// @name LLVM-style RTTI methods
  /// {

  Kind getKind() const { return kind; }

  template <typename T>
  bool isa() const {
    return llvm::isa<T>(this);
  }

  template <typename T>
  T *cast() {
    assert(isa<T>());
    return llvm::cast<T>(this);
  }

  template <typename T>
  const T *cast() const {
    assert(isa<T>());
    return llvm::cast<T>(this);
  }

  template <typename T>
  // NOLINTNEXTLINE(readability-identifier-naming)
  T *dyn_cast() {
    return llvm::dyn_cast<T>(this);
  }

  template <typename T>
  // NOLINTNEXTLINE(readability-identifier-naming)
  const T *dyn_cast() const {
    return llvm::dyn_cast<T>(this);
  }

  /// }

  virtual std::unique_ptr<ASTNode> clone() const = 0;

  SourceRange getLocation() const;

  void setLocation(SourceRange loc);

  ASTNode *getParent();

  const ASTNode *getParent() const;

  void setParent(ASTNode *node);

  template <typename T>
  T *getParentOfType() {
    const ASTNode *node = parent;

    while (node != nullptr) {
      if (T *casted = node->dyn_cast<T>()) {
        return casted;
      }

      node = node->parent;
    }

    return nullptr;
  }

  template <typename T>
  const T *getParentOfType() const {
    const ASTNode *node = parent;

    while (node != nullptr) {
      if (const T *casted = node->dyn_cast<T>()) {
        return casted;
      }

      node = node->parent;
    }

    return nullptr;
  }

  virtual llvm::json::Value toJSON() const = 0;

protected:
  virtual void addJSONProperties(llvm::json::Object &obj) const;

private:
  Kind kind;
  SourceRange location;
  ASTNode *parent;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_ASTNODE_H
