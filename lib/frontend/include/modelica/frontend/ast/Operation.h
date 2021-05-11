#pragma once

#include <initializer_list>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <modelica/utils/SourcePosition.h>
#include <utility>
#include <variant>
#include <vector>

#include "Expression.h"

namespace modelica::frontend
{
	enum class OperationKind
	{
		negate,
		add,
		subtract,
		multiply,
		divide,
		ifelse,
		greater,
		greaterEqual,
		equal,
		different,
		lessEqual,
		less,
		land,
		lor,
		subscription,
		memberLookup,
		powerOf,
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const OperationKind& obj);

	std::string toString(OperationKind operation);

	class Operation
			: public impl::ExpressionCRTP<Operation>,
				public impl::Cloneable<Operation>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Expression>>::iterator;
		using const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

		/*
		template<typename... Args>
		Operation(SourcePosition location, OperationKind kind, Args&&... args)
				: Expression(location, OPERATION),
					arguments({ std::forward<Args>(args)... }),
					kind(kind)
		{
		}
		 */

		Operation(SourcePosition location,
							OperationKind kind,
							llvm::ArrayRef<std::unique_ptr<Expression>> args);

		Operation(const Operation& other);
		Operation(Operation&& other);
		~Operation() override;

		Operation& operator=(const Operation& other);
		Operation& operator=(Operation&& other);

		friend void swap(Operation& first, Operation& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::EXPRESSION_OPERATION;
		}

		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t nestLevel = 0) const override;

		[[nodiscard]] bool isLValue() const override;

		[[nodiscard]] bool operator==(const Operation& other) const;
		[[nodiscard]] bool operator!=(const Operation& other) const;

		[[nodiscard]] Expression* operator[](size_t index);
		[[nodiscard]] const Expression* operator[](size_t index) const;

		[[nodiscard]] OperationKind getOperationKind() const;
		void setOperationKind(OperationKind k);

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Expression>> getArguments();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Expression>> getArguments() const;

		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		OperationKind kind;
		Container<std::unique_ptr<Expression>> args;
	};

	template<OperationKind op, typename... Args>
	[[nodiscard]] Operation makeOp(Args&&... args)
	{
		return Operation(op, std::forward(args)...);
	}

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Operation& obj);

	std::string toString(const Operation& obj);
}
