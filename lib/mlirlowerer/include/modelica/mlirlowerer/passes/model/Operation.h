#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/SmallVector.h>
#include <memory>

namespace modelica::codegen::model
{
	class Expression;

	class Operation
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;

		public:
		using iterator = boost::indirect_iterator<Container<Expression>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<Expression>::const_iterator>;

		Operation(llvm::ArrayRef<Expression> args);

		Expression& operator[](size_t index);
		const Expression& operator[](size_t index) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] size_t childrenCount() const;

		private:
		Container<Expression> args;
	};
}