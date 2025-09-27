#include "marco/Codegen/Lowering/BaseModelica/ClassPath.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"

namespace marco::codegen::lowering::bmodelica {
ClassPath::Scope::Scope(ClassPath &path)
    : path(&path), size(path.path.size()) {}

ClassPath::Scope::~Scope() {
  if (path->path.size() > size) {
    path->path.erase(std::next(path->path.begin(), size), path->path.end());
  }
}

bool ClassPath::operator==(const ClassPath &other) const {
  return path == other.path;
}

bool ClassPath::operator!=(const ClassPath &other) const {
  return path != other.path;
}

bool ClassPath::operator<(const ClassPath &other) const {
  if (path.size() < other.size()) {
    return true;
  }

  if (path.size() > other.size()) {
    return false;
  }

  for (size_t i = 0, e = path.size(); i < e; ++i) {
    if (path[i] < other.path[i]) {
      return true;
    }
  }

  return false;
}

llvm::ArrayRef<const ast::bmodelica::Class *> ClassPath::get() const {
  return path;
}

size_t ClassPath::size() const { return path.size(); }

void ClassPath::append(const ast::bmodelica::Class &cls) {
  path.push_back(&cls);
}

const ast::bmodelica::Class &ClassPath::getLeaf() const {
  assert(!path.empty());
  return *path.back();
}
} // namespace marco::codegen::lowering::bmodelica
