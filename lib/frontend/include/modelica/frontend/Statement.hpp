#pragma once

#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <vector>

#include "Expression.hpp"

namespace modelica
{
	class Statement;
	using UniqueStatement = std::unique_ptr<Statement>;

	class AssignmentStatement
	{
		public:
		AssignmentStatement(Expression destination, Expression expression);
		AssignmentStatement(Tuple destinations, Expression expression);
		AssignmentStatement(
				std::initializer_list<Expression> destinations, Expression expression);

		template<typename Iter>
		AssignmentStatement(
				Iter destinationsBegin, Iter destinationsEnd, Expression expression)
				: destination(Tuple(destinationsBegin, destinationsEnd)),
					expression(std::move(expression))
		{
		}

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents) const;

		[[nodiscard]] std::vector<Expression*> getDestinations();
		void setDestination(Expression destination);
		void setDestination(Tuple destinations);

		[[nodiscard]] Expression& getExpression();
		[[nodiscard]] const Expression& getExpression() const;

		private:
		template<typename T>
		[[nodiscard]] bool destinationIsA() const
		{
			return std::holds_alternative<T>(destination);
		}

		template<typename T>
		[[nodiscard]] T& getDestination()
		{
			assert(destinationIsA<T>());
			return std::get<T>(destination);
		}

		// Where the result of the expression has to be stored.
		// A vector is needed because functions may have multiple outputs.
		std::variant<Expression, Tuple> destination;

		// Right-hand side expression of the assignment
		Expression expression;
	};

	class ForStatement
	{
		public:
		ForStatement() {}

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;
	};

	class IfBlock
	{
		public:
		IfBlock(Expression condition, llvm::ArrayRef<Statement> statements);
		explicit IfBlock(llvm::ArrayRef<Statement> statements);

		IfBlock(const IfBlock& other);
		IfBlock(IfBlock&& other) = default;

		IfBlock& operator=(const IfBlock& other);
		IfBlock& operator=(IfBlock&& other) = default;

		~IfBlock() = default;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		private:
		Expression condition;
		llvm::SmallVector<UniqueStatement, 3> statements;
	};

	class IfStatement
	{
		public:
		explicit IfStatement(llvm::ArrayRef<IfBlock> blocks);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		private:
		llvm::SmallVector<IfBlock, 3> blocks;
	};

	class Statement
	{
		public:
		Statement(AssignmentStatement statement);
		Statement(ForStatement statement);
		Statement(IfStatement statement);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		template<class Visitor>
		auto visit(Visitor&& vis)
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		template<class Visitor>
		auto visit(Visitor&& vis) const
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		private:
		std::variant<AssignmentStatement, ForStatement, IfStatement> content;
	};

}	 // namespace modelica
