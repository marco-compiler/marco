#include "marco/AST/Node/ReferenceAccess.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ReferenceAccess::ReferenceAccess(SourceRange location)
      : Expression(
          ASTNode::Kind::Expression_ReferenceAccess, std::move(location)),
        dummy(false),
        globalLookup(false)
  {
  }

  ReferenceAccess::ReferenceAccess(const ReferenceAccess& other)
      : Expression(other),
        dummy(other.dummy),
        globalLookup(other.globalLookup)
  {
    setPathVariables(other.path);
  }

  ReferenceAccess::~ReferenceAccess() = default;

  std::unique_ptr<ASTNode> ReferenceAccess::clone() const
  {
    return std::make_unique<ReferenceAccess>(*this);
  }

  llvm::json::Value ReferenceAccess::toJSON() const
  {
    llvm::json::Object result;

    result["dummy"] = isDummy();
    result["globalLookup"] = isGlobalLookup();

    llvm::SmallVector<llvm::json::Value> pathJson;

    for (const auto& variable : path) {
      pathJson.emplace_back(variable);
    }

    result["path"] = llvm::json::Array(pathJson);

    addJSONProperties(result);
    return result;
  }

  bool ReferenceAccess::isLValue() const
  {
    return true;
  }

  bool ReferenceAccess::isDummy() const
  {
    return dummy;
  }

  void ReferenceAccess::setDummy(bool value)
  {
    dummy = value;
  }

  bool ReferenceAccess::isGlobalLookup() const
  {
    return globalLookup;
  }

  void ReferenceAccess::setGlobalLookup(bool global)
  {
    globalLookup = global;
  }

  llvm::ArrayRef<std::string> ReferenceAccess::getPathVariables() const
  {
    return path;
  }

  void ReferenceAccess::setPathVariables(llvm::ArrayRef<std::string> newPath)
  {
    path.clear();
    path.append(newPath.begin(), newPath.end());
  }

  std::string ReferenceAccess::getName() const
  {
    std::string result = "";

    if (globalLookup) {
      result += ".";
    }

    for (size_t i = 0, e = path.size(); i< e; ++i) {
      result += path[i];

      if (i != e - 1) {
        result += ".";
      }
    }

    return result;
  }
}
