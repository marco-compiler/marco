#include "marco/AST/Node/Record.h"
#include "marco/AST/Node/Member.h"
#include "marco/AST/Node/Type.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
RecordType::RecordType(llvm::ArrayRef<std::unique_ptr<ASTNode>> types) {
  for (const auto &type : types) {
    body.push_back(type->clone());
  }
}

size_t RecordType::size() const { return body.size(); }

const VariableType *RecordType::getType(size_t index) const {
  assert(index < body.size());
  return body[index]->cast<VariableType>();
}

Record::Record(SourceRange location)
    : Class(ASTNode::Kind::Class_Record, std::move(location)) {}

std::unique_ptr<ASTNode> Record::clone() const {
  return std::make_unique<Record>(*this);
}

llvm::json::Value Record::toJSON() const {
  llvm::json::Object result;
  addJSONProperties(result);
  return result;
}

RecordType Record::getType() const {
  llvm::SmallVector<std::unique_ptr<ASTNode>> types;

  for (const auto &variable : getVariables()) {
    types.push_back(variable->cast<Member>()->getType()->clone());
  }

  return RecordType(types);
}
} // namespace marco::ast
