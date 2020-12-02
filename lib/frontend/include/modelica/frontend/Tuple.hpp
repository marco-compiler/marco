#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace modelica
{
	class Expression;

	class Tuple
	{
		private:
		using UniqueExpression = std::unique_ptr<Expression>;
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = boost::indirect_iterator<Container<UniqueExpression>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<UniqueExpression>::const_iterator>;

		Tuple();
		explicit Tuple(std::initializer_list<Expression> expressions);
		explicit Tuple(llvm::ArrayRef<Expression> expressions);

		template<typename Iter>
		Tuple(Iter begin, Iter end)
		{
			for (auto it = begin; it != end; it++)
				expressions.push_back(std::make_unique<Expression>(*it));
		}

		Tuple(const Tuple& other);
		Tuple(Tuple&& other) = default;

		~Tuple() = default;

		Tuple& operator=(const Tuple& other);
		Tuple& operator=(Tuple&& other) = default;

		[[nodiscard]] bool operator==(const Tuple& other) const;
		[[nodiscard]] bool operator!=(const Tuple& other) const;

		[[nodiscard]] Expression& operator[](size_t index);
		[[nodiscard]] const Expression& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin()const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Container<UniqueExpression> expressions;
	};
}	 // namespace modelica
