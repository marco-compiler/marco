#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/ast/nodes/ASTNode.h"
#include "marco/ast/nodes/Type.h"
#include <initializer_list>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace marco::ast
{
	class Expression;

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
			: public ASTNode,
				public impl::Dumpable<Operation>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Expression>>::iterator;
		using const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

		Operation(const Operation& other);
		Operation(Operation&& other);
		~Operation() override;

		Operation& operator=(const Operation& other);
		Operation& operator=(Operation&& other);

		friend void swap(Operation& first, Operation& second);

		void print(llvm::raw_ostream& OS = llvm::outs(), size_t nestLevel = 0) const override;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] bool operator==(const Operation& other) const;
		[[nodiscard]] bool operator!=(const Operation& other) const;

		[[nodiscard]] Expression* operator[](size_t index);
		[[nodiscard]] const Expression* operator[](size_t index) const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		[[nodiscard]] OperationKind getOperationKind() const;
		void setOperationKind(OperationKind k);

		[[nodiscard]] Expression* getArg(size_t index);
		[[nodiscard]] const Expression* getArg(size_t index) const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Expression>> getArguments();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Expression>> getArguments() const;

		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		friend class Expression;

		Operation(SourceRange location,
							Type type,
							OperationKind kind,
							llvm::ArrayRef<std::unique_ptr<Expression>> args);

		Type type;
		OperationKind kind;
		Container<std::unique_ptr<Expression>> args;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Operation& obj);

	std::string toString(const Operation& obj);
}
