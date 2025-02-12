#include "marco/VariableFilter/AST.h"

using namespace ::marco;
using namespace ::marco::vf;

namespace marco::vf {
//===----------------------------------------------------------------------===//
// ASTNode
//===----------------------------------------------------------------------===//

ASTNode::~ASTNode() = default;

//===----------------------------------------------------------------------===//
// VariableExpression
//===----------------------------------------------------------------------===//

VariableExpression::VariableExpression(llvm::StringRef identifier)
    : identifier(identifier.str()) {}

llvm::StringRef VariableExpression::getIdentifier() const { return identifier; }

//===----------------------------------------------------------------------===//
// ArrayExpression
//===----------------------------------------------------------------------===//

ArrayExpression::ArrayExpression(VariableExpression variable,
                                 llvm::ArrayRef<Range> ranges)
    : variable(std::move(variable)), ranges(ranges.begin(), ranges.end()) {}

VariableExpression ArrayExpression::getVariable() const { return variable; }

llvm::ArrayRef<Range> ArrayExpression::getRanges() const { return ranges; }

//===----------------------------------------------------------------------===//
// DerivativeExpression
//===----------------------------------------------------------------------===//

DerivativeExpression::DerivativeExpression(VariableExpression derivedVariable)
    : derivedVariable(std::move(derivedVariable)) {}

VariableExpression DerivativeExpression::getDerivedVariable() const {
  return derivedVariable;
}

//===----------------------------------------------------------------------===//
// RegexExpression
//===----------------------------------------------------------------------===//

RegexExpression::RegexExpression(llvm::StringRef regex) : regex(regex.str()) {}

llvm::StringRef RegexExpression::getRegex() const { return regex; }
} // namespace marco::vf
