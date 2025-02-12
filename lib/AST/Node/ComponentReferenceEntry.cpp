#include "marco/AST/Node/ComponentReferenceEntry.h"
#include "marco/AST/Node/Subscript.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
ComponentReferenceEntry::ComponentReferenceEntry(SourceRange location)
    : ASTNode(ASTNode::Kind::ComponentReferenceEntry, std::move(location)) {}

ComponentReferenceEntry::ComponentReferenceEntry(
    const ComponentReferenceEntry &other)
    : ASTNode(other), name(other.name) {
  setSubscripts(other.subscripts);
}

ComponentReferenceEntry::~ComponentReferenceEntry() = default;

std::unique_ptr<ASTNode> ComponentReferenceEntry::clone() const {
  return std::make_unique<ComponentReferenceEntry>(*this);
}

llvm::json::Value ComponentReferenceEntry::toJSON() const {
  llvm::json::Object result;

  result["name"] = getName();

  llvm::SmallVector<llvm::json::Value> subscriptsJson;

  for (const auto &subscript : subscripts) {
    subscriptsJson.push_back(subscript->toJSON());
  }

  result["subscripts"] = llvm::json::Array(subscriptsJson);

  addJSONProperties(result);
  return result;
}

llvm::StringRef ComponentReferenceEntry::getName() const { return name; }

void ComponentReferenceEntry::setName(llvm::StringRef newName) {
  name = newName.str();
}

size_t ComponentReferenceEntry::getNumOfSubscripts() const {
  return subscripts.size();
}

Subscript *ComponentReferenceEntry::getSubscript(size_t index) {
  assert(index < subscripts.size());
  return subscripts[index]->cast<Subscript>();
}

const Subscript *ComponentReferenceEntry::getSubscript(size_t index) const {
  assert(index < subscripts.size());
  return subscripts[index]->cast<Subscript>();
}

void ComponentReferenceEntry::setSubscripts(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  subscripts.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Subscript>());
    auto &clone = subscripts.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast
