#ifndef MARCO_AST_ASTNODE_H
#define MARCO_AST_ASTNODE_H

#include "marco/Lexer/Location.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"

namespace llvm {
class raw_ostream;
}

namespace marco::ast {
class ASTNode {
public:
  template <typename Kind>
  ASTNode(Kind kind, SourceRange location, ASTNode *parent = nullptr)
      : kind(static_cast<int32_t>(kind)), location(std::move(location)),
        parent(parent) {}

  ASTNode(const ASTNode &other);

  virtual ~ASTNode();

  /// @name LLVM-style RTTI methods
  /// {

  template <typename Kind = int32_t>
  Kind getKind() const {
    return static_cast<Kind>(kind);
  }

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
  T *dyn_cast() {
    return llvm::dyn_cast<T>(this);
  }

  template <typename T>
  const T *dyn_cast() const {
    return llvm::dyn_cast<T>(this);
  }

  /// }

  virtual std::unique_ptr<ast::ASTNode> clone() const = 0;

  SourceRange getLocation() const { return location; }

  void setLocation(SourceRange loc) { this->location = std::move(loc); }

  ASTNode *getParent() { return parent; }

  const ASTNode *getParent() const { return parent; }

  void setParent(ASTNode *node) { parent = node; }

  template <typename T>
  T *getParentOfType() {
    ASTNode *node = parent;

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
    ASTNode *node = parent;

    while (node != nullptr) {
      if (T *casted = node->dyn_cast<T>()) {
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
  const int32_t kind;
  SourceRange location;
  ASTNode *parent;
};
} // namespace marco::ast

#endif // MARCO_AST_ASTNODE_H
