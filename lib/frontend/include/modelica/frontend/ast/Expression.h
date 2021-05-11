#pragma once

#include <initializer_list>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <utility>

#include "ASTNode.h"
#include "Type.h"

namespace modelica::frontend
{
	class Expression : public impl::ASTNodeCRTP<Expression>
	{
		public:
		Expression(ASTNodeKind kind, SourcePosition location, Type type);

		~Expression() override;

		friend void swap(Expression& first, Expression& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::EXPRESSION &&
						 node->getKind() <= ASTNodeKind::EXPRESSION_LAST_EXPRESSION;
		}

		[[nodiscard]] virtual std::unique_ptr<Expression> cloneExpression() const = 0;

		[[nodiscard]] virtual bool operator==(const Expression& other) const = 0;
		[[nodiscard]] bool operator!=(const Expression& other) const;

		[[nodiscard]] virtual bool isLValue() const = 0;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		/*
		template<typename Arg>
		[[nodiscard]] static Constant constant(SourcePosition location, Type type, Arg&& arg)
		{
			Constant content(std::move(location), std::forward<Arg>(arg));
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression reference(SourcePosition location, Type type, Args&&... args)
		{
			ReferenceAccess content(location, std::forward<Args>(args)...);
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression operation(SourcePosition location, Type type, OperationKind kind, Args&&... args)
		{
			Operation content(location, kind, std::forward<Args>(args)...);
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression call(SourcePosition location, Type type, Expression function, Args&&... args)
		{
			Call content(location, std::move(function), { std::forward<Args>(args)... });
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression tuple(SourcePosition location, Type type, Args&&... args)
		{
			Tuple content(location, { std::forward<Args>(args)... });
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression array(SourcePosition location, Type type, Args&&... args)
		{
			Array content(location, { std::forward<Args>(args)... });
			return Expression(type, std::move(content));
		}
		 */

		protected:
		Expression(const Expression& other);
		Expression(Expression&& other);

		Expression& operator=(const Expression& other);
		Expression& operator=(Expression&& other);

		private:
		Type type;
	};

	namespace impl
	{
		template<typename Derived>
		struct ExpressionCRTP : public Expression
		{
			using Expression::Expression;

			[[nodiscard]] bool operator==(const Expression& other) const override
			{
				if (auto* casted = other.template dyn_cast<Derived>())
					return static_cast<const Derived&>(*this) == *casted;

				return false;
			}

			[[nodiscard]] std::unique_ptr<Expression> cloneExpression() const override
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}
		};
	}

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Expression& obj);

	std::string toString(const Expression& obj);
}
