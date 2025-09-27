#include "marco/AST/ASTNode.h"

using namespace ::marco::ast;

namespace marco::ast {
ASTNode::ASTNode(const marco::ast::ASTNode &other) = default;

ASTNode::~ASTNode() = default;

void ASTNode::addJSONProperties(llvm::json::Object &obj) const {}
} // namespace marco::ast
