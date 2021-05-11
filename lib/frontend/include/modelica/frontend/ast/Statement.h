#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <stack>
#include <vector>

#include "ASTNode.h"
#include "ConditionalBlock.h"

namespace modelica::frontend
{
	class AssignmentStatement;
	class BreakStatement;
	class Expression;
	class ForStatement;
	class IfStatement;
	class ReturnStatement;
	class Tuple;
	class WhenStatement;
	class WhileStatement;

	template<typename ValueType, typename NodeType>
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

		value_type* operator->()
		{
			return current;
		}

		private:
		void fetchNext()
		{
			bool found = false;

			while (!found && !statements.empty())
			{
				auto& statement = statements.top();
				statements.pop();

				auto visitor = [&](NodeType* statement) -> llvm::Optional<Statement*> {
					if (auto& assignmentStatement = llvm::dyn_cast<AssignmentStatement>(statement))
						return &assignmentStatement;

					if (auto& ifStatement = llvm::dyn_cast<IfStatement>(statement))
					{
						for (auto i = ifStatement.size(); i > 0; i--)
						{
							auto& block = ifStatement[i - 1];

							for (auto j = block.size(); j > 0; j--)
								statements->push(block[j - 1]);
						}

						return llvm::None;
					}

					if (auto& forStatement = llvm::dyn_cast<ForStatement>(statement))
					{
						for (auto i = forStatement.size(); i > 0; i--)
							statements->push(forStatement[i - 1]);

						return llvm::None;
					}

					if (auto& whileStatement = llvm::dyn_cast<WhileStatement>(statement))
					{
						for (auto i = whileStatement.size(); i > 0; i--)
							statements->push(whileStatement[i - 1]);

						return llvm::None;
					}

					if (auto& whenStatement = llvm::dyn_cast<WhenStatement>(statement))
					{
						for (auto i = whenStatement.size(); i > 0; i--)
							statements->push(whenStatement[i - 1]);

						return llvm::None;
					}

					if (auto& breakStatement = llvm::dyn_cast<BreakStatement>(statement))
						return llvm::None;

					if (auto& returnStatement = llvm::dyn_cast<ReturnStatement>(statement))
						return llvm::None;
				};

				auto assignment = visitor(statement);

				if (assignment.hasValue())
				{
					current = assignment.getValue();
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

	class Statement : public impl::ASTNodeCRTP<Statement>
	{
		public:
		using assignments_iterator = AssignmentsIterator<
				AssignmentStatement, Statement>;

		using assignments_const_iterator = AssignmentsIterator<
		    const AssignmentStatement, const Statement>;

		Statement(ASTNodeKind kind, SourcePosition location);

		~Statement() override;

		friend void swap(Statement& first, Statement& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::STATEMENT &&
						 node->getKind() <= ASTNodeKind::STATEMENT_LAST;
		}

		[[nodiscard]] virtual std::unique_ptr<Statement> cloneStatement() const = 0;

		[[nodiscard]] assignments_iterator assignmentsBegin();
		[[nodiscard]] assignments_const_iterator assignmentsBegin() const;

		[[nodiscard]] assignments_iterator assignmentsEnd();
		[[nodiscard]] assignments_const_iterator assignmentsEnd() const;

		protected:
		Statement(const Statement& other);
		Statement(Statement&& other);

		Statement& operator=(const Statement& other);
		Statement& operator=(Statement&& other);
	};

	namespace impl
	{
		template<typename Derived>
		struct StatementCRTP : public Statement
		{
			using Statement::Statement;

			[[nodiscard]] std::unique_ptr<Statement> cloneStatement() const override
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}
		};
	}

	class AssignmentStatement
			: public impl::StatementCRTP<AssignmentStatement>,
				public impl::Cloneable<AssignmentStatement>
	{
		public:
		AssignmentStatement(SourcePosition location,
												std::unique_ptr<Expression> destination,
												std::unique_ptr<Expression> expression);

		AssignmentStatement(SourcePosition location,
												std::unique_ptr<Tuple> destinations,
												std::unique_ptr<Expression> expression);

		/*
		AssignmentStatement(
				SourcePosition location, std::initializer_list<Expression> destinations, Expression expression);
				*/

		/*
		template<typename Iter>
		AssignmentStatement(
				SourcePosition location, Iter destinationsBegin, Iter destinationsEnd, Expression expression)
				: location(std::move(location)),
					destinations(Tuple(destinationsBegin, destinationsEnd)),
					expression(std::move(expression))
		{
		}
		 */

		AssignmentStatement(const AssignmentStatement& other);
		AssignmentStatement(AssignmentStatement&& other);
		~AssignmentStatement() override;

		AssignmentStatement& operator=(const AssignmentStatement& other);
		AssignmentStatement& operator=(AssignmentStatement&& other);

		friend void swap(AssignmentStatement& first, AssignmentStatement& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::STATEMENT_ASSIGNMENT;
		}

		void dump(llvm::raw_ostream& os, size_t indents) const override;

		[[nodiscard]] Tuple* getDestinations();
		[[nodiscard]] const Tuple* getDestinations() const;

		void setDestinations(Expression* destination);
		void setDestinations(Tuple* destinations);

		[[nodiscard]] Expression* getExpression();
		[[nodiscard]] const Expression* getExpression() const;

		private:
		// Where the result of the expression has to be stored.
		// A tuple is needed because functions may have multiple outputs.
		std::unique_ptr<Tuple> destinations;

		// Right-hand side expression of the assignment
		std::unique_ptr<Expression> expression;
	};

	class IfStatement
			: public impl::StatementCRTP<IfStatement>,
				public impl::Cloneable<IfStatement>
	{
		public:
		using Block = ConditionalBlock<Statement>;

		private:
		using Container = llvm::SmallVector<Block, 3>;

		public:
		using blocks_iterator = Container::iterator;
		using blocks_const_iterator = Container::const_iterator;

		IfStatement(SourcePosition location, llvm::ArrayRef<Block> blocks);

		IfStatement(const IfStatement& other);
		IfStatement(IfStatement&& other);
		~IfStatement() override;

		IfStatement& operator=(const IfStatement& other);
		IfStatement& operator=(IfStatement&& other);

		friend void swap(IfStatement& first, IfStatement& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::STATEMENT_ASSIGNMENT;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Block& operator[](size_t index);
		[[nodiscard]] const Block& operator[](size_t index) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] blocks_iterator begin();
		[[nodiscard]] blocks_const_iterator begin() const;

		[[nodiscard]] blocks_iterator end();
		[[nodiscard]] blocks_const_iterator end() const;

		private:
		Container blocks;
	};

	class ForStatement
			: public impl::StatementCRTP<ForStatement>,
				public impl::Cloneable<ForStatement>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using statements_iterator = Container<std::unique_ptr<Statement>>::iterator;
		using statements_const_iterator = Container<std::unique_ptr<Statement>>::const_iterator;

		ForStatement(SourcePosition location,
								 std::unique_ptr<Induction>& induction,
								 llvm::ArrayRef<std::unique_ptr<Statement>> statements);

		ForStatement(const ForStatement& other);
		ForStatement(ForStatement&& other);
		~ForStatement() override;

		ForStatement& operator=(const ForStatement& other);
		ForStatement& operator=(ForStatement&& other);

		friend void swap(ForStatement& first, ForStatement& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::STATEMENT_FOR;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Statement& operator[](size_t index);
		[[nodiscard]] const Statement& operator[](size_t index) const;

		[[nodiscard]] llvm::StringRef getBreakCheckName() const;
		void setBreakCheckName(llvm::StringRef name);

		[[nodiscard]] llvm::StringRef getReturnCheckName() const;
		void setReturnCheckName(llvm::StringRef name);

		[[nodiscard]] Induction* getInduction();
		[[nodiscard]] const Induction* getInduction() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Statement>> getBody();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Statement>> getBody() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] statements_iterator begin();
		[[nodiscard]] statements_const_iterator begin() const;

		[[nodiscard]] statements_iterator end();
		[[nodiscard]] statements_const_iterator end() const;

		private:
		std::unique_ptr<Induction> induction;
		Container<std::unique_ptr<Statement>> statements;
		std::string breakCheckName;
		std::string returnCheckName;
	};

	class WhileStatement
			: public impl::StatementCRTP<WhileStatement>,
				public impl::Cloneable<WhileStatement>,
				public ConditionalBlock<Statement>
	{
		public:
		WhileStatement(SourcePosition location,
									 std::unique_ptr<Expression>& condition,
									 llvm::ArrayRef<std::unique_ptr<Statement>> body);

		WhileStatement(const WhileStatement& other);
		WhileStatement(WhileStatement&& other);
		~WhileStatement() override;

		WhileStatement& operator=(const WhileStatement& other);
		WhileStatement& operator=(WhileStatement&& other);

		friend void swap(WhileStatement& first, WhileStatement& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::STATEMENT_WHILE;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] llvm::StringRef getBreakCheckName() const;
		void setBreakCheckName(llvm::StringRef name);

		[[nodiscard]] llvm::StringRef getReturnCheckName() const;
		void setReturnCheckName(llvm::StringRef name);

		private:
		std::string breakCheckName;
		std::string returnCheckName;
	};

	class WhenStatement
			: public impl::StatementCRTP<WhenStatement>,
				public impl::Cloneable<WhenStatement>,
				public ConditionalBlock<Statement>
	{
		public:
		WhenStatement(SourcePosition location,
									std::unique_ptr<Expression>& condition,
									llvm::ArrayRef<std::unique_ptr<Statement>> body);

		WhenStatement(const WhenStatement& other);
		WhenStatement(WhenStatement&& other);
		~WhenStatement() override;

		WhenStatement& operator=(const WhenStatement& other);
		WhenStatement& operator=(WhenStatement&& other);

		friend void swap(WhenStatement& first, WhenStatement& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::STATEMENT_WHEN;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;
	};

	class BreakStatement
			: public impl::StatementCRTP<BreakStatement>,
				public impl::Cloneable<BreakStatement>
	{
		public:
		explicit BreakStatement(SourcePosition location);

		BreakStatement(const BreakStatement& other);
		BreakStatement(BreakStatement&& other);
		~BreakStatement() override;

		BreakStatement& operator=(const BreakStatement& other);
		BreakStatement& operator=(BreakStatement&& other);

		friend void swap(BreakStatement& first, BreakStatement& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::STATEMENT_BREAK;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;
	};

	class ReturnStatement
			: public impl::StatementCRTP<ReturnStatement>,
				public impl::Cloneable<ReturnStatement>
	{
		public:
		explicit ReturnStatement(SourcePosition location);

		ReturnStatement(const ReturnStatement& other);
		ReturnStatement(ReturnStatement&& other);
		~ReturnStatement() override;

		ReturnStatement& operator=(const ReturnStatement& other);
		ReturnStatement& operator=(ReturnStatement&& other);

		friend void swap(ReturnStatement& first, ReturnStatement& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::STATEMENT_RETURN;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] llvm::StringRef getReturnCheckName() const;
		void setReturnCheckName(llvm::StringRef name);

		private:
		std::string returnCheckName;
	};
}
