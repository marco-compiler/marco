#ifndef MARCO_AST_NODE_CONSTANT_H
#define MARCO_AST_NODE_CONSTANT_H

#include "marco/AST/Node/Expression.h"
#include <string>
#include <variant>

namespace marco::ast {
class Constant : public Expression {
public:
  Constant(SourceRange location);

  Constant(const Constant &other);

  ~Constant() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Expression_Constant;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool isLValue() const override;

  template <class Visitor>
  auto visit(Visitor &&vis) {
    return std::visit(std::forward<Visitor>(vis), value);
  }

  template <class Visitor>
  auto visit(Visitor &&vis) const {
    return std::visit(std::forward<Visitor>(vis), value);
  }

  template <typename T>
  T as() const {
    if constexpr (std::is_same_v<T, std::string>) {
      if (std::holds_alternative<std::string>(value)) {
        return std::get<std::string>(value);
      }

      if (std::holds_alternative<bool>(value)) {
        return std::get<bool>(value) ? "true" : "false";
      }

      if (std::holds_alternative<int64_t>(value)) {
        return std::to_string(std::get<int64_t>(value));
      }

      return std::to_string(std::get<double>(value));
    } else if constexpr (std::is_same_v<T, bool> || std::is_integral_v<T>) {
      if (std::holds_alternative<bool>(value)) {
        return static_cast<T>(std::get<bool>(value));
      }

      if (std::holds_alternative<int64_t>(value)) {
        return static_cast<T>(std::get<int64_t>(value));
      }

      return static_cast<T>(!std::get<std::string>(value).empty());
    } else {
      if (std::holds_alternative<bool>(value)) {
        return static_cast<T>(std::get<bool>(value));
      }

      if (std::holds_alternative<int64_t>(value)) {
        return static_cast<T>(std::get<int64_t>(value));
      }

      if (std::holds_alternative<double>(value)) {
        return static_cast<T>(std::get<double>(value));
      }

      return static_cast<T>(!std::get<std::string>(value).empty());
    }
  }

  void setValue(bool newValue);
  void setValue(int64_t newValue);
  void setValue(double newValue);
  void setValue(std::string newValue);

  // Utility methods for tests.
  void setValue(int32_t newValue);
  void setValue(float newValue);

private:
  std::variant<bool, int64_t, double, std::string> value;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_CONSTANT_H
