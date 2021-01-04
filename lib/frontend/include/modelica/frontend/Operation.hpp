#pragma once

#include <initializer_list>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <modelica/utils/SourceRange.hpp>
#include <utility>
#include <variant>
#include <vector>

namespace modelica
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

	class Expression;

	class Operation
	{
		public:
		using Container = std::vector<Expression>;
		using iterator = Container::iterator;
		using const_iterator = Container::const_iterator;

		template<typename... Args>
		Operation(SourcePosition location, OperationKind kind, Args&&... args)
				: location(std::move(location)),
					arguments({ std::forward<Args>(args)... }),
					kind(kind)
		{
		}

		Operation(SourcePosition location, OperationKind kind, Container args);

		[[nodiscard]] bool operator==(const Operation& other) const;
		[[nodiscard]] bool operator!=(const Operation& other) const;

		[[nodiscard]] Expression& operator[](size_t index);
		[[nodiscard]] const Expression& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t nestLevel = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] OperationKind getKind() const;
		void setKind(OperationKind k);

		[[nodiscard]] Container& getArguments();
		[[nodiscard]] const Container& getArguments() const;
		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		SourcePosition location;
		std::vector<Expression> arguments;
		OperationKind kind;
	};

	template<OperationKind op, typename... Args>
	[[nodiscard]] Operation makeOp(Args&&... args)
	{
		return Operation(op, std::forward(args)...);
	}
}	 // namespace modelica
