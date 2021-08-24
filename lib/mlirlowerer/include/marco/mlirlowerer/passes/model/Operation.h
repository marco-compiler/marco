#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <memory>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

namespace marco::codegen::model
{
	class Expression;

	class Operation
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::shared_ptr<Expression>>::iterator;
		using const_iterator = Container<std::shared_ptr<Expression>>::const_iterator;

		Operation(llvm::ArrayRef<Expression> args);

		std::shared_ptr<Expression> operator[](size_t index);
		std::shared_ptr<Expression> operator[](size_t index) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Container<std::shared_ptr<Expression>> args;
	};
}