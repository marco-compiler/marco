#pragma once

#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <stack>
#include <vector>

#include "Expression.hpp"
#include "Induction.hpp"

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
		ForStatement(Induction induction, llvm::ArrayRef<Statement> statements);

		ForStatement(const ForStatement& other);
		ForStatement(ForStatement&& other) = default;

		ForStatement& operator=(const ForStatement& other);
		ForStatement& operator=(ForStatement&& other) = default;

		~ForStatement() = default;

		[[nodiscard]] UniqueStatement& operator[](size_t index);
		[[nodiscard]] const UniqueStatement& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] Induction& getInduction();
		[[nodiscard]] const Induction& getInduction() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] llvm::SmallVectorImpl<UniqueStatement>::iterator begin();
		[[nodiscard]] llvm::SmallVectorImpl<UniqueStatement>::const_iterator begin()
				const;

		[[nodiscard]] llvm::SmallVectorImpl<UniqueStatement>::iterator end();
		[[nodiscard]] llvm::SmallVectorImpl<UniqueStatement>::const_iterator end()
				const;

		private:
		Induction induction;
		llvm::SmallVector<UniqueStatement, 3> statements;
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

		[[nodiscard]] UniqueStatement& operator[](size_t index);
		[[nodiscard]] const UniqueStatement& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] Expression& getCondition();
		[[nodiscard]] const Expression& getCondition() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] llvm::SmallVectorImpl<UniqueStatement>::iterator begin();
		[[nodiscard]] llvm::SmallVectorImpl<UniqueStatement>::const_iterator begin()
				const;

		[[nodiscard]] llvm::SmallVectorImpl<UniqueStatement>::iterator end();
		[[nodiscard]] llvm::SmallVectorImpl<UniqueStatement>::const_iterator end()
				const;

		private:
		Expression condition;
		llvm::SmallVector<UniqueStatement, 3> statements;
	};

	class IfStatement
	{
		public:
		explicit IfStatement(llvm::ArrayRef<IfBlock> blocks);

		[[nodiscard]] IfBlock& operator[](size_t index);
		[[nodiscard]] const IfBlock& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] llvm::SmallVectorImpl<IfBlock>::iterator begin();
		[[nodiscard]] llvm::SmallVectorImpl<IfBlock>::const_iterator begin() const;

		[[nodiscard]] llvm::SmallVectorImpl<IfBlock>::iterator end();
		[[nodiscard]] llvm::SmallVectorImpl<IfBlock>::const_iterator end() const;

		private:
		llvm::SmallVector<IfBlock, 3> blocks;
	};

	class AssignmentsIterator
	{
		public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = AssignmentStatement;
		using difference_type = AssignmentStatement;
		using pointer = AssignmentStatement*;
		using reference = AssignmentStatement&;

		AssignmentsIterator(Statement* root, Statement* start);

		operator bool() const;

		bool operator==(const AssignmentsIterator& it) const;
		bool operator!=(const AssignmentsIterator& it) const;

		AssignmentsIterator& operator++();
		AssignmentsIterator operator++(int);

		value_type& operator*();
		const value_type& operator*() const;

		private:
		void fetchNext();

		Statement* root;
		std::stack<Statement*> statements;
		AssignmentStatement* next;
	};

	class Statement
	{
		public:
		using iterator = AssignmentsIterator;

		Statement(AssignmentStatement statement);
		Statement(ForStatement statement);
		Statement(IfStatement statement);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] iterator end();

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
