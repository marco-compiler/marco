#include "marco/AST/BaseModelica/Node/FunctionArgument.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
FunctionArgument::FunctionArgument(const FunctionArgument &other) = default;

FunctionArgument::~FunctionArgument() = default;

void FunctionArgument::addJSONProperties(llvm::json::Object &obj) const {
  ASTNode::addJSONProperties(obj);
}
} // namespace marco::ast::bmodelica
