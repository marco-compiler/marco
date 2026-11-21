#include "marco/AST/Modelica/MemberAccess.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void MemberAccess::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(base, callback);
}

void MemberAccess::addJSONProperties(llvm::json::Object &obj) const {
  obj["base"] = modelica::toJSON(base.get());
  obj["member"] = member;
}

Expression *MemberAccess::getBase() const { return base.get(); }

void MemberAccess::setBase(std::unique_ptr<Expression> node) {
  base = std::move(node);
}

llvm::StringRef MemberAccess::getMember() const { return member; }

void MemberAccess::setMember(std::string name) { member = std::move(name); }
} // namespace marco::ast::modelica
