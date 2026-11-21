#ifndef MARCO_AST_MODELICA_CHILD_H
#define MARCO_AST_MODELICA_CHILD_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::ast::modelica {
/// Owning pointer to a child node, which is deep-copied when copied.
template <typename T>
class Child {
public:
  Child() = default;

  Child(std::unique_ptr<T> node) : node(std::move(node)) {}

  Child(const Child &other)
      : node(other.node ? std::unique_ptr<T>(
                              static_cast<T *>(other.node->clone().release()))
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

/// Default container of the lists of child nodes.
template <typename T>
using DefaultChildrenStorage = llvm::SmallVector<T>;

/// Owning list of child nodes. The container can be changed through
/// 'Storage', which is instantiated with the type of the elements.
template <typename T,
          template <typename> class Storage = DefaultChildrenStorage>
using Children = Storage<Child<T>>;

/// Read-only view of a list of child nodes, whose container must store the
/// elements contiguously.
template <typename T>
using ChildrenRef = llvm::ArrayRef<Child<T>>;
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_CHILD_H
