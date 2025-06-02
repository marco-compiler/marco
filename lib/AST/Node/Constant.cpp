#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Constant.h"
#include "marco/AST/Node/Expression.h"
#include "marco/Parser/Location.h"

#include <llvm/Support/JSON.h>

#include <cstdint>
#include <memory>
#include <string>

#include <utility>

#include <variant>

using namespace ::marco;
using namespace ::marco::ast;

namespace {
struct ConstantJsonVisitor {
  llvm::json::Value operator()(bool value) { return llvm::json::Value(value); }

  llvm::json::Value operator()(int64_t value) {
    return llvm::json::Value(value);
  }

  llvm::json::Value operator()(double value) {
    return llvm::json::Value(value);
  }

  llvm::json::Value operator()(std::string value) {
    return llvm::json::Value(std::move(value));
  }
};
} // namespace

namespace marco::ast {
Constant::Constant(SourceRange location)
    : Expression(ASTNode::Kind::Expression_Constant, std::move(location)) {}

Constant::Constant(const Constant &other)
     = default;

Constant::~Constant() = default;

std::unique_ptr<ASTNode> Constant::clone() const {
  return std::make_unique<Constant>(*this);
}

llvm::json::Value Constant::toJSON() const {
  llvm::json::Object result;

  ConstantJsonVisitor visitor;
  result["value"] = std::visit(visitor, value);

  addJSONProperties(result);
  return result;
}

bool Constant::isLValue() const { return false; }

void Constant::setValue(bool newValue) { value = newValue; }

void Constant::setValue(int64_t newValue) { value = newValue; }

void Constant::setValue(double newValue) { value = newValue; }

void Constant::setValue(std::string newValue) { value = newValue; }

void Constant::setValue(int32_t newValue) {
  value = static_cast<int64_t>(newValue);
}

void Constant::setValue(float newValue) {
  value = static_cast<double>(newValue);
}
} // namespace marco::ast
