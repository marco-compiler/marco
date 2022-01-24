#pragma once

#include "boost/iterator/indirect_iterator.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/ast/nodes/ASTNode.h"
#include "marco/ast/nodes/ConditionalBlock.h"
#include <stack>
#include <vector>

namespace marco::ast
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

	class AssignmentStatement
			: public ASTNode,
				public impl::Dumpable<AssignmentStatement>
	{
		public:
		AssignmentStatement(const AssignmentStatement& other);
		AssignmentStatement(AssignmentStatement&& other);
		~AssignmentStatement() override;

		AssignmentStatement& operator=(const AssignmentStatement& other);
		AssignmentStatement& operator=(AssignmentStatement&& other);

		friend void swap(AssignmentStatement& first, AssignmentStatement& second);

		void print(llvm::raw_ostream& os, size_t indents) const override;

		[[nodiscard]] Expression* getDestinations();
		[[nodiscard]] const Expression* getDestinations() const;

		void setDestinations(std::unique_ptr<Expression> destination);

		[[nodiscard]] Expression* getExpression();
		[[nodiscard]] const Expression* getExpression() const;

		private:
		friend class Statement;

		AssignmentStatement(SourceRange location,
												std::unique_ptr<Expression> destination,
												std::unique_ptr<Expression> expression);

		// Where the result of the expression has to be stored.
		// It is always a tuple, because functions may have multiple outputs.
		std::unique_ptr<Expression> destinations;

		// Right-hand side expression of the assignment
		std::unique_ptr<Expression> expression;
	};

	class BreakStatement
			: public ASTNode,
				public impl::Dumpable<BreakStatement>
	{
		public:
		BreakStatement(const BreakStatement& other);
		BreakStatement(BreakStatement&& other);
		~BreakStatement() override;

		BreakStatement& operator=(const BreakStatement& other);
		BreakStatement& operator=(BreakStatement&& other);

		friend void swap(BreakStatement& first, BreakStatement& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		private:
		friend class Statement;

		explicit BreakStatement(SourceRange location);
	};

	class ForStatement
			: public ASTNode,
				public impl::Dumpable<ForStatement>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using statements_iterator = Container<std::unique_ptr<Statement>>::iterator;
		using statements_const_iterator = Container<std::unique_ptr<Statement>>::const_iterator;

		ForStatement(const ForStatement& other);
		ForStatement(ForStatement&& other);
		~ForStatement() override;

		ForStatement& operator=(const ForStatement& other);
		ForStatement& operator=(ForStatement&& other);

		friend void swap(ForStatement& first, ForStatement& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Statement* operator[](size_t index);
		[[nodiscard]] const Statement* operator[](size_t index) const;

		[[nodiscard]] llvm::StringRef getBreakCheckName() const;
		void setBreakCheckName(llvm::StringRef name);

		[[nodiscard]] llvm::StringRef getReturnCheckName() const;
		void setReturnCheckName(llvm::StringRef name);

		[[nodiscard]] Induction* getInduction();
		[[nodiscard]] const Induction* getInduction() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Statement>> getBody();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Statement>> getBody() const;

		void setBody(llvm::ArrayRef<std::unique_ptr<Statement>> body);

		[[nodiscard]] size_t size() const;

		[[nodiscard]] statements_iterator begin();
		[[nodiscard]] statements_const_iterator begin() const;

		[[nodiscard]] statements_iterator end();
		[[nodiscard]] statements_const_iterator end() const;

		private:
		friend class Statement;

		ForStatement(SourceRange location,
								 std::unique_ptr<Induction> induction,
								 llvm::ArrayRef<std::unique_ptr<Statement>> statements);

		std::unique_ptr<Induction> induction;
		Container<std::unique_ptr<Statement>> statements;
		std::string breakCheckName;
		std::string returnCheckName;
	};

	class IfStatement :
			public ASTNode,
			public impl::Dumpable<IfStatement>
	{
		public:
		using Block = ConditionalBlock<Statement>;

		private:
		using Container = llvm::SmallVector<Block, 3>;

		public:
		using blocks_iterator = Container::iterator;
		using blocks_const_iterator = Container::const_iterator;

		IfStatement(const IfStatement& other);
		IfStatement(IfStatement&& other);
		~IfStatement() override;

		IfStatement& operator=(const IfStatement& other);
		IfStatement& operator=(IfStatement&& other);

		friend void swap(IfStatement& first, IfStatement& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Block& operator[](size_t index);
		[[nodiscard]] const Block& operator[](size_t index) const;

		[[nodiscard]] Block& getBlock(size_t index);
		[[nodiscard]] const Block& getBlock(size_t index) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] blocks_iterator begin();
		[[nodiscard]] blocks_const_iterator begin() const;

		[[nodiscard]] blocks_iterator end();
		[[nodiscard]] blocks_const_iterator end() const;

		private:
		friend class Statement;

		IfStatement(SourceRange location, llvm::ArrayRef<Block> blocks);

		Container blocks;
	};

	class ReturnStatement
			: public ASTNode,
				public impl::Dumpable<ReturnStatement>
	{
		public:
		ReturnStatement(const ReturnStatement& other);
		ReturnStatement(ReturnStatement&& other);
		~ReturnStatement() override;

		ReturnStatement& operator=(const ReturnStatement& other);
		ReturnStatement& operator=(ReturnStatement&& other);

		friend void swap(ReturnStatement& first, ReturnStatement& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] llvm::StringRef getReturnCheckName() const;
		void setReturnCheckName(llvm::StringRef name);

		private:
		friend class Statement;

		explicit ReturnStatement(SourceRange location);

		std::string returnCheckName;
	};

	class WhenStatement
			: public ASTNode,
				public ConditionalBlock<Statement>,
				public impl::Dumpable<WhenStatement>
	{
		public:
		WhenStatement(const WhenStatement& other);
		WhenStatement(WhenStatement&& other);
		~WhenStatement() override;

		WhenStatement& operator=(const WhenStatement& other);
		WhenStatement& operator=(WhenStatement&& other);

		friend void swap(WhenStatement& first, WhenStatement& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		private:
		friend class Statement;

		WhenStatement(SourceRange location,
									std::unique_ptr<Expression> condition,
									llvm::ArrayRef<std::unique_ptr<Statement>> body);
	};

	class WhileStatement
			: public ASTNode,
				public ConditionalBlock<Statement>,
				public impl::Dumpable<WhileStatement>
	{
		public:
		WhileStatement(const WhileStatement& other);
		WhileStatement(WhileStatement&& other);
		~WhileStatement() override;

		WhileStatement& operator=(const WhileStatement& other);
		WhileStatement& operator=(WhileStatement&& other);

		friend void swap(WhileStatement& first, WhileStatement& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Statement* operator[](size_t index);
		[[nodiscard]] const Statement* operator[](size_t index) const;

		[[nodiscard]] llvm::StringRef getBreakCheckName() const;
		void setBreakCheckName(llvm::StringRef name);

		[[nodiscard]] llvm::StringRef getReturnCheckName() const;
		void setReturnCheckName(llvm::StringRef name);

		private:
		friend class Statement;

		WhileStatement(SourceRange location,
									 std::unique_ptr<Expression> condition,
									 llvm::ArrayRef<std::unique_ptr<Statement>> body);

		std::string breakCheckName;
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
					statements->push(block.getBody()[j - 1].get());
			}

			return nullptr;
		}

		ValueType* operator()(
				std::tuple_element_t<2, std::tuple<Variants...>>& forStatement)
		{
			for (auto i = forStatement.size(); i > 0; i--)
				statements->push(forStatement.getBody()[i - 1].get());

			return nullptr;
		}

		ValueType* operator()(
				std::tuple_element_t<3, std::tuple<Variants...>>& whileStatement)
		{
			for (auto i = whileStatement.size(); i > 0; i--)
				statements->push(whileStatement.getBody()[i - 1].get());

			return nullptr;
		}

		ValueType* operator()(
				std::tuple_element_t<4, std::tuple<Variants...>>& whenStatement)
		{
			for (auto i = whenStatement.size(); i > 0; i--)
				statements->push(whenStatement.getBody()[i - 1].get());

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
			: public impl::Cloneable<Statement>,
				public impl::Dumpable<Statement>
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
		Statement(BreakStatement statement);
		Statement(ForStatement statement);
		Statement(IfStatement statement);
		Statement(ReturnStatement statement);
		Statement(WhenStatement statement);
		Statement(WhileStatement statement);

		Statement(const Statement& other);
		Statement(Statement&& other);

		~Statement();

		Statement& operator=(const Statement& other);
		Statement& operator=(Statement&& other);

		friend void swap(Statement& first, Statement& second);

		void print(llvm::raw_ostream &os, size_t indents = 0) const override;

		template<typename T>
		[[nodiscard]] bool isa() const
		{
			return std::holds_alternative<T>(content);
		}

		template<typename T>
		[[nodiscard]] T* get()
		{
			assert(isa<T>());
			return &std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T* get() const
		{
			assert(isa<T>());
			return &std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] T* dyn_get()
		{
			if (!isa<T>())
				return nullptr;

			return get<T>();
		}

		template<typename T>
		[[nodiscard]] const T* dyn_get() const
		{
			if (!isa<T>())
				return nullptr;

			return get<T>();
		}

		template<typename Visitor>
		auto visit(Visitor&& visitor)
		{
			return std::visit(visitor, content);
		}

		template<typename Visitor>
		auto visit(Visitor&& visitor) const
		{
			return std::visit(visitor, content);
		}

		[[nodiscard]] SourceRange getLocation() const;

		[[nodiscard]] assignments_iterator begin();
		[[nodiscard]] assignments_const_iterator begin() const;

		[[nodiscard]] assignments_iterator end();
		[[nodiscard]] assignments_const_iterator end() const;

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Statement> assignmentStatement(Args&&... args)
		{
				return std::make_unique<Statement>(AssignmentStatement(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Statement> breakStatement(Args&&... args)
		{
				return std::make_unique<Statement>(BreakStatement(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Statement> forStatement(Args&&... args)
		{
				return std::make_unique<Statement>(ForStatement(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Statement> ifStatement(Args&&... args)
		{
				return std::make_unique<Statement>(IfStatement(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Statement> returnStatement(Args&&... args)
		{
			return std::make_unique<Statement>(ReturnStatement(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Statement> whenStatement(Args&&... args)
		{
			return std::make_unique<Statement>(WhenStatement(std::forward<Args>(args)...));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Statement> whileStatement(Args&&... args)
		{
			return std::make_unique<Statement>(WhileStatement(std::forward<Args>(args)...));
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
}
