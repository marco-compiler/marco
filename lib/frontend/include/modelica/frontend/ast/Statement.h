#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <stack>
#include <vector>

#include "ConditionalBlock.h"
#include "Expression.h"
#include "Induction.h"

namespace modelica
{
	class Statement;

	class AssignmentStatement
	{
		public:
		AssignmentStatement(SourcePosition location, Expression destination, Expression expression);
		AssignmentStatement(SourcePosition location, Tuple destinations, Expression expression);
		AssignmentStatement(
				SourcePosition location, std::initializer_list<Expression> destinations, Expression expression);

		template<typename Iter>
		AssignmentStatement(
				SourcePosition location, Iter destinationsBegin, Iter destinationsEnd, Expression expression)
				: location(std::move(location)),
					destinations(Tuple(destinationsBegin, destinationsEnd)),
					expression(std::move(expression))
		{
		}

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] Tuple& getDestinations();
		[[nodiscard]] const Tuple& getDestinations() const;

		void setDestination(Expression destination);
		void setDestination(Tuple destinations);

		[[nodiscard]] Expression& getExpression();
		[[nodiscard]] const Expression& getExpression() const;

		private:
		SourcePosition location;

		// Where the result of the expression has to be stored.
		// A tuple is needed because functions may have multiple outputs.
		Tuple destinations;

		// Right-hand side expression of the assignment
		Expression expression;
	};

	class IfStatement
	{
		public:
		using Block = ConditionalBlock<Statement>;

		private:
		using Container = llvm::SmallVector<Block, 3>;

		public:
		using blocks_iterator = Container::iterator;
		using blocks_const_iterator = Container::const_iterator;

		IfStatement(SourcePosition location, llvm::ArrayRef<Block> blocks);

		[[nodiscard]] Block& operator[](size_t index);
		[[nodiscard]] const Block& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] blocks_iterator begin();
		[[nodiscard]] blocks_const_iterator begin() const;

		[[nodiscard]] blocks_iterator end();
		[[nodiscard]] blocks_const_iterator end() const;

		private:
		SourcePosition location;
		Container blocks;
	};

	class ForStatement
	{
		private:
		using UniqueStatement = std::shared_ptr<Statement>;
		using Container = llvm::SmallVector<UniqueStatement, 3>;

		public:
		using statements_iterator = Container::iterator;
		using statements_const_iterator = Container::const_iterator;

		ForStatement(SourcePosition location, Induction induction, llvm::ArrayRef<Statement> statements);

		ForStatement(const ForStatement& other);
		ForStatement(ForStatement&& other) = default;

		ForStatement& operator=(const ForStatement& other);
		ForStatement& operator=(ForStatement&& other) = default;

		~ForStatement() = default;

		[[nodiscard]] Statement& operator[](size_t index);
		[[nodiscard]] const Statement& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] const std::string& getBreakCheckName() const;
		void setBreakCheckName(std::string name);

		[[nodiscard]] const std::string& getReturnCheckName() const;
		void setReturnCheckName(std::string name);

		[[nodiscard]] Induction& getInduction();
		[[nodiscard]] const Induction& getInduction() const;

		[[nodiscard]] Container& getBody();
		[[nodiscard]] const Container& getBody() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] statements_iterator begin();
		[[nodiscard]] statements_const_iterator begin() const;

		[[nodiscard]] statements_iterator end();
		[[nodiscard]] statements_const_iterator end() const;

		private:
		SourcePosition location;
		Induction induction;
		Container statements;
		std::string breakCheckName;
		std::string returnCheckName;
	};

	class WhileStatement: public ConditionalBlock<Statement>
	{
		public:
		WhileStatement(SourcePosition location, Expression condition, llvm::ArrayRef<Statement> body);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] const std::string& getBreakCheckName() const;
		void setBreakCheckName(std::string name);

		[[nodiscard]] const std::string& getReturnCheckName() const;
		void setReturnCheckName(std::string name);

		private:
		SourcePosition location;
		std::string breakCheckName;
		std::string returnCheckName;
	};

	class WhenStatement: public ConditionalBlock<Statement>
	{
		public:
		WhenStatement(SourcePosition location, Expression condition, llvm::ArrayRef<Statement> body);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		private:
		SourcePosition location;
	};

	class BreakStatement
	{
		public:
		explicit BreakStatement(SourcePosition location);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] const std::string& getBreakCheckName() const;
		void setBreakCheckName(std::string name);

		private:
		SourcePosition location;
		std::string breakCheckName;
	};

	class ReturnStatement
	{
		public:
		explicit ReturnStatement(SourcePosition location);

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] const std::string& getReturnCheckName() const;
		void setReturnCheckName(std::string name);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		private:
		SourcePosition location;
		std::string returnCheckName;
	};

	template<typename ValueType, typename NodeType, typename... Variants>
	class AssignmentsIteratorVisitor
	{
		public:
		AssignmentsIteratorVisitor(std::stack<NodeType*>* statements)
				: statements(statements){};

		ValueType* operator()(
				std::tuple_element_t<0, std::tuple<Variants...>>& statement)
		{
			return &statement;
		}

		ValueType* operator()(
				std::tuple_element_t<1, std::tuple<Variants...>>& ifStatement)
		{
			for (auto i = ifStatement.size(); i > 0; i--)
			{
				auto& block = ifStatement[i - 1];

				for (auto j = block.size(); j > 0; j--)
					statements->push(&block[j - 1]);
			}

			return nullptr;
		}

		ValueType* operator()(
				std::tuple_element_t<2, std::tuple<Variants...>>& forStatement)
		{
			for (auto i = forStatement.size(); i > 0; i--)
				statements->push(&forStatement[i - 1]);

			return nullptr;
		}

		ValueType* operator()(
				std::tuple_element_t<3, std::tuple<Variants...>>& whileStatement)
		{
			for (auto i = whileStatement.size(); i > 0; i--)
				statements->push(&whileStatement[i - 1]);

			return nullptr;
		}

		ValueType* operator()(
				std::tuple_element_t<4, std::tuple<Variants...>>& whenStatement)
		{
			for (auto i = whenStatement.size(); i > 0; i--)
				statements->push(&whenStatement[i - 1]);

			return nullptr;
		}

		ValueType* operator()(
				std::tuple_element_t<5, std::tuple<Variants...>>& whenStatement)
		{
			return nullptr;
		}

		ValueType* operator()(
				std::tuple_element_t<6, std::tuple<Variants...>>& whenStatement)
		{
			return nullptr;
		}

		private:
		std::stack<NodeType*>* statements;
	};

	template<
			typename ValueType,
			typename NodeType,
			template<typename, typename, typename...>
			class Visitor,
			typename... Variants>
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
				auto* assignment = statement->visit(
						Visitor<ValueType, NodeType, Variants...>(&statements));

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
		using assignments_iterator = AssignmentsIterator<
				AssignmentStatement,
				Statement,
				AssignmentsIteratorVisitor,
				AssignmentStatement,
				IfStatement,
				ForStatement,
				WhileStatement,
				WhenStatement,
				BreakStatement,
				ReturnStatement>;

		using assignments_const_iterator = AssignmentsIterator<
				const AssignmentStatement,
				const Statement,
				AssignmentsIteratorVisitor,
				const AssignmentStatement,
				const IfStatement,
				const ForStatement,
				const WhileStatement,
				const WhenStatement,
				const BreakStatement,
				const ReturnStatement>;

		Statement(AssignmentStatement statement);
		Statement(IfStatement statement);
		Statement(ForStatement statement);
		Statement(WhileStatement statement);
		Statement(WhenStatement statement);
		Statement(BreakStatement statement);
		Statement(ReturnStatement statement);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T>(content);
		}

		template<typename T>
		[[nodiscard]] T& get()
		{
			assert(isA<T>());
			return std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(isA<T>());
			return std::get<T>(content);
		}

		[[nodiscard]] assignments_iterator begin();
		[[nodiscard]] assignments_const_iterator begin() const;

		[[nodiscard]] assignments_iterator end();
		[[nodiscard]] assignments_const_iterator end() const;

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
				WhenStatement,
				BreakStatement,
				ReturnStatement>
				content;
	};

}	 // namespace modelica
