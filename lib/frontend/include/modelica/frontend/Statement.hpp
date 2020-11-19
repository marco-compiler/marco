#pragma once

#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <stack>
#include <vector>

#include "ConditionalBlock.hpp"
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
		// Where the result of the expression has to be stored.
		// A vector is needed because functions may have multiple outputs.
		std::variant<Expression, Tuple> destination;

		// Right-hand side expression of the assignment
		Expression expression;
	};

	class IfStatement
	{
		public:
		using Block = ConditionalBlock<Statement>;

		explicit IfStatement(llvm::ArrayRef<Block> blocks);

		[[nodiscard]] Block& operator[](size_t index);
		[[nodiscard]] const Block& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] llvm::SmallVectorImpl<Block>::iterator begin();
		[[nodiscard]] llvm::SmallVectorImpl<Block>::const_iterator begin() const;

		[[nodiscard]] llvm::SmallVectorImpl<Block>::iterator end();
		[[nodiscard]] llvm::SmallVectorImpl<Block>::const_iterator end() const;

		private:
		llvm::SmallVector<Block, 3> blocks;
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

	class WhileStatement: public ConditionalBlock<Statement>
	{
		public:
		WhileStatement(Expression condition, llvm::ArrayRef<Statement> body);
	};

	class WhenStatement: public ConditionalBlock<Statement>
	{
		public:
		WhenStatement(Expression condition, llvm::ArrayRef<Statement> body);
	};

	class AssignmentsIteratorVisitor
	{
		public:
		AssignmentsIteratorVisitor(std::stack<Statement*>* statements);

		AssignmentStatement* operator()(AssignmentStatement& statement);
		AssignmentStatement* operator()(IfStatement& ifStatement);
		AssignmentStatement* operator()(ForStatement& forStatement);
		AssignmentStatement* operator()(WhileStatement& whileStatement);
		AssignmentStatement* operator()(WhenStatement& whenStatement);

		private:
		std::stack<Statement*>* statements;
	};

	class AssignmentsConstIteratorVisitor
	{
		public:
		AssignmentsConstIteratorVisitor(std::stack<const Statement*>* statements);

		const AssignmentStatement* operator()(const AssignmentStatement& statement);
		const AssignmentStatement* operator()(const IfStatement& ifStatement);
		const AssignmentStatement* operator()(const ForStatement& forStatement);
		const AssignmentStatement* operator()(const WhileStatement& whileStatement);
		const AssignmentStatement* operator()(const WhenStatement& whenStatement);

		private:
		std::stack<const Statement*>* statements;
	};

	template<typename ValueType, typename NodeType, class Visitor>
	class AssignmentsIterator
	{
		public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = ValueType;
		using difference_type = std::ptrdiff_t;
		using pointer = ValueType*;
		using reference = ValueType&;

		AssignmentsIterator(): AssignmentsIterator(nullptr, nullptr) {}

		AssignmentsIterator(NodeType* root, NodeType* start): root(root)
		{
			if (start != nullptr)
				statements.push(start);

			fetchNext();
		}

		operator bool() const { return !statements.empty(); }

		bool operator==(const AssignmentsIterator& it) const
		{
			return root == it.root && statements.size() == it.statements.size() &&
						 current == it.current;
		}

		bool operator!=(const AssignmentsIterator& it) const
		{
			return !(*this == it);
		}

		AssignmentsIterator& operator++()
		{
			fetchNext();
			return *this;
		}

		AssignmentsIterator operator++(int)
		{
			auto temp = *this;
			fetchNext();
			return temp;
		}

		value_type& operator*()
		{
			assert(current != nullptr);
			return *current;
		}

		const value_type& operator*() const
		{
			assert(current != nullptr);
			return *current;
		}

		value_type* operator->() { return current; }

		private:
		void fetchNext()
		{
			bool found = false;

			while (!found && !statements.empty())
			{
				auto& statement = statements.top();
				statements.pop();
				auto* assignment = statement->visit(Visitor(&statements));

				if (assignment != nullptr)
				{
					current = assignment;
					found = true;
				}
			}

			if (!found)
				current = nullptr;
		}

		NodeType* root;
		std::stack<NodeType*> statements;
		value_type* current;
	};

	class Statement
	{
		public:
		using iterator = AssignmentsIterator<
				AssignmentStatement,
				Statement,
				AssignmentsIteratorVisitor>;
		using const_iterator = AssignmentsIterator<
				const AssignmentStatement,
				const Statement,
				AssignmentsConstIteratorVisitor>;

		Statement(AssignmentStatement statement);
		Statement(ForStatement statement);
		Statement(IfStatement statement);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator cbegin();
		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator cend();

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
		std::variant<
				AssignmentStatement,
				IfStatement,
				ForStatement,
				WhileStatement,
				WhenStatement>
				content;
	};

}	 // namespace modelica
