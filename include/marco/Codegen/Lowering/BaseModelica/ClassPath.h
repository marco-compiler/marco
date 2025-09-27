#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_CLASSPATH_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_CLASSPATH_H

#include "marco/AST/BaseModelica/AST.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace marco::codegen::lowering::bmodelica {
class ClassPath {
public:
  class Scope {
  public:
    Scope(ClassPath &path);

    ~Scope();

  private:
    ClassPath *path;
    size_t size;
  };

  bool operator==(const ClassPath &other) const;

  bool operator!=(const ClassPath &other) const;

  bool operator<(const ClassPath &other) const;

  llvm::ArrayRef<const ast::bmodelica::Class *> get() const;

  size_t size() const;

  void append(const ast::bmodelica::Class &cls);

  const ast::bmodelica::Class &getLeaf() const;

private:
  std::vector<const ast::bmodelica::Class *> path;
};
} // namespace marco::codegen::lowering::bmodelica

namespace llvm {
template <>
struct DenseMapInfo<marco::codegen::lowering::bmodelica::ClassPath> {
  static inline marco::codegen::lowering::bmodelica::ClassPath getEmptyKey() {
    return {};
  }

  static inline marco::codegen::lowering::bmodelica::ClassPath
  getTombstoneKey() {
    return {};
  }

  static unsigned
  getHashValue(const marco::codegen::lowering::bmodelica::ClassPath &val) {
    return llvm::hash_value(val.get());
  }

  static bool
  isEqual(const marco::codegen::lowering::bmodelica::ClassPath &lhs,
          const marco::codegen::lowering::bmodelica::ClassPath &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_CLASSPATH_H
