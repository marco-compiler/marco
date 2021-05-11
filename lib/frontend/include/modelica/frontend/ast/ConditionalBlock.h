#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

#include "Expression.h"

namespace modelica::frontend
{
	template<typename T>
	class ConditionalBlock
	{
		private:
		using Container = llvm::SmallVector<std::unique_ptr<T>, 3>;

		public:
		using iterator = typename Container::iterator;
		using const_iterator = typename Container::const_iterator;

		ConditionalBlock(std::unique_ptr<Expression>& condition,
										 llvm::ArrayRef<std::unique_ptr<T>> body)
				: condition(condition->cloneExpression())
		{
			for (const auto& element : body)
				this->body.push_back(element->cloneStatement());
		}

		ConditionalBlock(const ConditionalBlock<T>& other)
				: condition(other.condition->cloneExpression())
		{
			for (const auto& element : body)
				this->body.push_back(element->cloneStatement());
		}

		ConditionalBlock(ConditionalBlock<T>&& other) = default;
		~ConditionalBlock() = default;

		ConditionalBlock<T>& operator=(const ConditionalBlock<T>& other)
		{
			ConditionalBlock<T> result(other);
			swap(*this, result);
			return *this;
		}

		ConditionalBlock& operator=(ConditionalBlock&& other) = default;

		friend void swap(ConditionalBlock<T>& first, ConditionalBlock<T>& second)
		{
			using std::swap;
			swap(first.breakCheckName, second.breakCheckName);
			swap(first.returnCheckName, second.returnCheckName);
		}

		[[nodiscard]] T& operator[](size_t index)
		{
			assert(index < body.size());
			return *body[index];
		}

		[[nodiscard]] const T& operator[](size_t index) const
		{
			assert(index < body.size());
			return *body[index];
		}

		void dump() const { dump(llvm::outs(), 0); }

		void dump(llvm::raw_ostream& os, size_t indents = 0) const
		{
			os.indent(indents);
			os << "condition:\n";
			condition->dump(os, indents + 1);

			os.indent(indents);
			os << "body:\n";

			for (const auto& statement : body)
				statement->dump(os, indents + 1);
		}

		[[nodiscard]] Expression* getCondition() { return condition.get(); }

		[[nodiscard]] const Expression* getCondition() const { return condition.get(); }

		[[nodiscard]] Container& getBody() { return body; }

		[[nodiscard]] const Container& getBody() const { return body; }

		[[nodiscard]] size_t size() const { return body.size(); }

		[[nodiscard]] iterator begin()
		{
			return body.begin();
		}

		[[nodiscard]] const_iterator begin() const
		{
			return body.begin();
		}

		[[nodiscard]] iterator end()
		{
			return body.end();
		}

		[[nodiscard]] const_iterator end() const
		{
			return body.end();
		}

		private:
		std::unique_ptr<Expression> condition;
		Container body;
	};
}
