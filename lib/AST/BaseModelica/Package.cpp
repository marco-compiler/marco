#include "marco/AST/BaseModelica/Package.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
Package::Package(SourceRange location)
    : Class(ASTNodeKind::Class_Package, std::move(location)) {}

std::unique_ptr<ASTNode> Package::clone() const {
  return std::make_unique<Package>(*this);
}

llvm::json::Value Package::toJSON() const {
  llvm::json::Object result;
  addNodeKindToJSON(*this, result);
  return result;
}
} // namespace marco::ast::bmodelica
