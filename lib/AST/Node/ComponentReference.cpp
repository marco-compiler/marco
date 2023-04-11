#include "marco/AST/Node/ComponentReference.h"
#include "marco/AST/Node/ComponentReferenceEntry.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ComponentReference::ComponentReference(SourceRange location)
      : Expression(
          ASTNode::Kind::Expression_ComponentReference, std::move(location)),
        dummy(false),
        globalLookup(false)
  {
  }

  ComponentReference::ComponentReference(const ComponentReference& other)
      : Expression(other),
        dummy(other.dummy),
        globalLookup(other.globalLookup)
  {
    setPath(other.path);
  }

  ComponentReference::~ComponentReference() = default;

  std::unique_ptr<ASTNode> ComponentReference::clone() const
  {
    return std::make_unique<ComponentReference>(*this);
  }

  llvm::json::Value ComponentReference::toJSON() const
  {
    llvm::json::Object result;

    result["dummy"] = isDummy();
    result["globalLookup"] = isGlobalLookup();

    llvm::SmallVector<llvm::json::Value> pathJson;

    for (const auto& element : path) {
      pathJson.emplace_back(element->toJSON());
    }

    result["path"] = llvm::json::Array(pathJson);

    addJSONProperties(result);
    return result;
  }

  bool ComponentReference::isLValue() const
  {
    return true;
  }

  bool ComponentReference::isDummy() const
  {
    return dummy;
  }

  void ComponentReference::setDummy(bool value)
  {
    dummy = value;
  }

  bool ComponentReference::isGlobalLookup() const
  {
    return globalLookup;
  }

  void ComponentReference::setGlobalLookup(bool global)
  {
    globalLookup = global;
  }

  size_t ComponentReference::getPathLength() const
  {
    return path.size();
  }

  ComponentReferenceEntry* ComponentReference::getElement(size_t index)
  {
    assert(index < path.size());
    return path[index]->cast<ComponentReferenceEntry>();
  }

  const ComponentReferenceEntry* ComponentReference::getElement(
      size_t index) const
  {
    assert(index < path.size());
    return path[index]->cast<ComponentReferenceEntry>();
  }

  void ComponentReference::setPath(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> newPath)
  {
    path.clear();

    for (const auto& node : newPath) {
      assert(node->isa<ComponentReferenceEntry>());
      auto& clone = path.emplace_back(node->clone());
      clone->setParent(this);
    }
  }

  std::string ComponentReference::getName() const
  {
    std::string result = "";

    if (globalLookup) {
      result += ".";
    }

    for (size_t i = 0, e = path.size(); i< e; ++i) {
      result += path[i]->cast<ComponentReferenceEntry>()->getName().str();

      if (i != e - 1) {
        result += ".";
      }
    }

    return result;
  }
}
