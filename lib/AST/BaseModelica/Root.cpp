#include "marco/AST/BaseModelica/Root.h"
#include "marco/AST/BaseModelica/Class.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
Root::Root(SourceRange location)
    : ASTNode(ASTNodeKind::Root, std::move(location)) {}

Root::Root(const Root &other) : ASTNode(other) {
  setInnerClasses(other.innerClasses);
}

Root::~Root() = default;

std::unique_ptr<ASTNode> Root::clone() const {
  return std::make_unique<Root>(*this);
}

llvm::json::Value Root::toJSON() const {
  llvm::json::Object result;

  llvm::SmallVector<llvm::json::Value> innerClassesJson;

  for (const auto &innerClass : innerClasses) {
    innerClassesJson.push_back(innerClass->toJSON());
  }

  result["inner_classes"] = llvm::json::Array(innerClassesJson);

  addNodeKindToJSON(*this, result);
  return result;
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Root::getInnerClasses() const {
  return innerClasses;
}

void Root::setInnerClasses(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> newInnerClasses) {
  innerClasses.clear();

  for (const auto &cls : newInnerClasses) {
    assert(cls->isa<Class>());
    auto &clone = innerClasses.emplace_back(cls->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast::bmodelica
