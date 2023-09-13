#ifndef MARCO_AST_NODE_COSTANT_H
#define MARCO_AST_NODE_COSTANT_H

#include "marco/AST/Node/Expression.h"
#include <string>
#include <variant>

namespace marco::ast
{
	class Constant : public Expression
	{
		public:
      Constant(SourceRange location);

      Constant(const Constant& other);

      ~Constant() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_Constant;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isLValue() const override;

      template<class Visitor>
      auto visit(Visitor&& vis)
      {
        return std::visit(std::forward<Visitor>(vis), value);
      }

      template<class Visitor>
      auto visit(Visitor&& vis) const
      {
        return std::visit(std::forward<Visitor>(vis), value);
      }

      template<typename T>
      T as() const
      {
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
}

#endif // MARCO_AST_NODE_COSTANT_H
