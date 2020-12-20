#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/frontend/Expression.hpp>

namespace modelica
{
	template<typename T>
	class ConditionalBlock
	{
		private:
		using uniqueType = std::unique_ptr<T>;
		using Container = llvm::SmallVector<uniqueType, 3>;

		public:
		using iterator = boost::indirect_iterator<typename Container::iterator>;
		using const_iterator = boost::indirect_iterator<typename Container::const_iterator>;

		ConditionalBlock(Expression condition, llvm::ArrayRef<T> body)
				: condition(std::move(condition))
		{
			for (const auto& element : body)
				this->body.emplace_back(std::make_unique<T>(element));
		}

		ConditionalBlock(const ConditionalBlock& other): condition(other.condition)
		{
			body.clear();

			for (const auto& element : other.body)
				body.emplace_back(std::make_unique<T>(*element));
		}

		ConditionalBlock(ConditionalBlock&& other) = default;

		ConditionalBlock& operator=(const ConditionalBlock& other)
		{
			if (this == &other)
				return *this;

			condition = other.condition;
			body.clear();

			for (const auto& element : other.body)
				body.emplace_back(std::make_unique<T>(*element));

			return *this;
		}

		ConditionalBlock& operator=(ConditionalBlock&& other) = default;

		~ConditionalBlock() = default;

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
			condition.dump(os, indents + 1);

			os.indent(indents);
			os << "body:\n";

			for (const auto& statement : body)
				statement->dump(os, indents + 1);
		}

		[[nodiscard]] Expression& getCondition() { return condition; }

		[[nodiscard]] const Expression& getCondition() const { return condition; }

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
		Expression condition;
		Container body;
	};
}	 // namespace modelica