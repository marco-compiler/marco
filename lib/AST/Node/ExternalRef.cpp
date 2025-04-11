using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
ExternalRef::ExternalRef(SourceRange location)
    : ASTNode(ASTNode::Kind::Algorithm, std::move(location)) {}

ExternalRef::ExternalRef(const ExternalRef &other)
    : ASTNode(other), languageSpecification(other.languageSpecification), externalFunctionCall(other.externalFunctionCall), annotationClause(other.annotationClause){
  setStatements(other.statements);
}

ExternalRef::~ExternalRef() = default;

std::unique_ptr<ASTNode> ExternalRef::clone() const {
  return std::make_unique<ExternalRef>(*this);
}

llvm::json::Value ExternalRef::toJSON() const {
  llvm::json::Object result;
  result["languageSpecification"] = languageSpecification;
  result["externalFunctionCall"] = externalFunctionCall -> toJSON();
  result["annotationClause"] = annotationClause -> toJSON();
  addJSONProperties(result);
  return result;

}

} // namespace marco::ast
