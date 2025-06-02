
#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Class.h"
#include "marco/AST/Node/Root.h"
#include "marco/Parser/Location.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>

#include <cassert>
#include <memory>
#include <utility>


using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
Root::Root(SourceRange location)
    : ASTNode(ASTNode::Kind::Root, std::move(location)) {}

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

  addJSONProperties(result);
  return result;
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Root::getInnerClasses() const {
  return innerClasses;
}

void Root::setInnerClasses(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  innerClasses.clear();

  for (const auto &cls : nodes) {
    assert(cls->isa<Class>());
    auto &clone = innerClasses.emplace_back(cls->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast
