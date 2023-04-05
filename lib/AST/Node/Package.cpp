#include "marco/AST/Node/Package.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Package::Package(SourceRange location)
      : Class(ASTNode::Kind::Class_Package, std::move(location))
  {
  }

  std::unique_ptr<ASTNode> Package::clone() const
  {
    return std::make_unique<Package>(*this);
  }

  llvm::json::Value Package::toJSON() const
  {
    llvm::json::Object result;
    addJSONProperties(result);
    return result;
  }
}
