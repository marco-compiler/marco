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
		using ExpressionPtr = std::shared_ptr<Expression>;
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = boost::indirect_iterator<Container<ExpressionPtr>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<ExpressionPtr>::const_iterator>;

		Operation(llvm::ArrayRef<Expression> args);

		ExpressionPtr operator[](size_t index);
		const ExpressionPtr operator[](size_t index) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] size_t childrenCount() const;

		private:
		Container<ExpressionPtr> args;
	};
}