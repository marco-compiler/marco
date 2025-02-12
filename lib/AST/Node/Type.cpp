#include "marco/AST/Node/Type.h"
#include "marco/AST/Node/ArrayDimension.h"

using namespace ::marco;
using namespace ::marco::ast;

static std::string toString(BuiltInType::Kind builtInTypeKind) {
  switch (builtInTypeKind) {
  case BuiltInType::Kind::Boolean:
    return "Boolean";
  case BuiltInType::Kind::Integer:
    return "Integer";
  case BuiltInType::Kind::Real:
    return "Real";
  case BuiltInType::Kind::String:
    return "String";
  default:
    llvm_unreachable("Unknown built-in type");
    return "unknown";
  }
}

namespace marco::ast {
VariableType::VariableType(const VariableType &other) : ASTNode(other) {
  setDimensions(other.dimensions);
}

VariableType::~VariableType() = default;

void VariableType::addJSONProperties(llvm::json::Object &obj) const {
  llvm::SmallVector<llvm::json::Value> dimensionsJson;

  for (const auto &dimension : dimensions) {
    dimensionsJson.push_back(dimension->toJSON());
  }

  obj["dimensions"] = llvm::json::Array(dimensionsJson);
  ASTNode::addJSONProperties(obj);
}

size_t VariableType::getRank() const { return dimensions.size(); }

ArrayDimension *VariableType::operator[](size_t index) {
  assert(index < dimensions.size());
  return dimensions[index]->cast<ArrayDimension>();
}

const ArrayDimension *VariableType::operator[](size_t index) const {
  assert(index < dimensions.size());
  return dimensions[index]->cast<ArrayDimension>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> VariableType::getDimensions() const {
  return dimensions;
}

void VariableType::setDimensions(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  dimensions.clear();

  for (const auto &node : nodes) {
    assert(node->isa<ArrayDimension>());
    auto &clone = dimensions.emplace_back(node->clone());
    clone->setParent(this);
  }
}

bool VariableType::hasConstantShape() const {
  return llvm::none_of(
      dimensions, [](const std::unique_ptr<ASTNode> &arrayDimension) {
        return arrayDimension->cast<ArrayDimension>()->isDynamic();
      });
}

bool VariableType::isScalar() const { return getRank() == 0; }

std::unique_ptr<ASTNode> VariableType::subscript(size_t times) const {
  assert(times <= getRank());
  auto result = clone();

  result->cast<VariableType>()->setDimensions(
      llvm::ArrayRef(dimensions).drop_front(times));

  return std::move(result);
}

BuiltInType::BuiltInType(SourceRange location)
    : VariableType(ASTNode::Kind::VariableType_BuiltIn, std::move(location)) {}

BuiltInType::BuiltInType(const BuiltInType &other)
    : VariableType(other), kind(other.kind) {}

BuiltInType::~BuiltInType() = default;

std::unique_ptr<ASTNode> BuiltInType::clone() const {
  return std::make_unique<BuiltInType>(*this);
}

llvm::json::Value BuiltInType::toJSON() const {
  llvm::json::Object result;
  result["type"] = toString(kind);

  addJSONProperties(result);
  return result;
}

BuiltInType::Kind BuiltInType::getBuiltInTypeKind() const { return kind; }

void BuiltInType::setBuiltInTypeKind(BuiltInType::Kind newKind) {
  kind = newKind;
}

bool BuiltInType::isNumeric() const {
  return kind == BuiltInType::Kind::Boolean ||
         kind == BuiltInType::Kind::Integer || kind == BuiltInType::Kind::Real;
}

UserDefinedType::UserDefinedType(SourceRange location)
    : VariableType(ASTNode::Kind::VariableType_UserDefined,
                   std::move(location)) {}

UserDefinedType::UserDefinedType(const UserDefinedType &other)
    : VariableType(other), globalLookup(other.globalLookup) {
  setPath(other.path);
}

UserDefinedType::~UserDefinedType() = default;

std::unique_ptr<ASTNode> UserDefinedType::clone() const {
  return std::make_unique<UserDefinedType>(*this);
}

llvm::json::Value UserDefinedType::toJSON() const {
  llvm::json::Object result;
  result["global_lookup"] = isGlobalLookup();

  llvm::SmallVector<llvm::json::Value> pathJson;

  for (const auto &name : path) {
    pathJson.emplace_back(name);
  }

  result["path"] = llvm::json::Array(pathJson);

  addJSONProperties(result);
  return result;
}

bool UserDefinedType::isGlobalLookup() const { return globalLookup; }

void UserDefinedType::setGlobalLookup(bool global) { globalLookup = global; }

size_t UserDefinedType::getPathLength() const { return path.size(); }

llvm::StringRef UserDefinedType::getElement(size_t index) const {
  assert(index < path.size());
  return path[index];
}

void UserDefinedType::setPath(llvm::ArrayRef<std::string> newPath) {
  path.clear();
  path.append(newPath.begin(), newPath.end());
}
} // namespace marco::ast
