#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/FunctionArgument.h"
#include <llvm/Support/JSON.h>


using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
FunctionArgument::FunctionArgument(const FunctionArgument &other) = default;

FunctionArgument::~FunctionArgument() = default;

void FunctionArgument::addJSONProperties(llvm::json::Object &obj) const {
  ASTNode::addJSONProperties(obj);
}
} // namespace marco::ast
