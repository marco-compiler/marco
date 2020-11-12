#pragma once

#include <initializer_list>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
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

		template<typename... Args>
		explicit Operation(OperationKind kind, Args&&... args)
				: arguments({ std::forward<Args>(args)... }), kind(kind)
		{
		}

		Operation(OperationKind kind, Container args);

		[[nodiscard]] bool operator==(const Operation& other) const;
		[[nodiscard]] bool operator!=(const Operation& other) const;

		[[nodiscard]] Expression& operator[](size_t index);
		[[nodiscard]] const Expression& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t nestLevel = 0) const;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] OperationKind getKind() const;
		void setKind(OperationKind k);

		[[nodiscard]] Container& getArguments();
		[[nodiscard]] const Container& getArguments() const;
		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] Container::iterator begin();
		[[nodiscard]] Container::const_iterator begin() const;

		[[nodiscard]] Container::iterator end();
		[[nodiscard]] Container::const_iterator end() const;

		private:
		Container arguments;
		OperationKind kind;
	};

	template<OperationKind op, typename... Args>
	[[nodiscard]] Operation makeOp(Args&&... args)
	{
		return Operation(op, std::forward(args)...);
	}
}	 // namespace modelica
