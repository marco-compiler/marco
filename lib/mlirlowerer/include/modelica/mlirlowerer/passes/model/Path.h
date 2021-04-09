#pragma once

#include <llvm/ADT/SmallVector.h>
#include <memory>

namespace modelica::codegen::model
{
	class Expression;

	class EquationPath
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T>;

		public:
		using iterator = Container<size_t>::iterator;
		using const_iterator = Container<size_t>::const_iterator;

		EquationPath(llvm::SmallVector<size_t, 3> path, bool left);

		[[nodiscard]] const_iterator begin() const;
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] size_t depth() const;
		[[nodiscard]] bool isOnEquationLeftHand() const;

		[[nodiscard]] Expression reach(Expression exp) const;

		private:
		Container<size_t> path;
		bool left;
	};

	class ExpressionPath
	{
		public:
		ExpressionPath(Expression exp, llvm::SmallVector<size_t, 3> path, bool left);
		ExpressionPath(Expression exp, EquationPath path);

		[[nodiscard]] EquationPath::const_iterator begin() const;
		[[nodiscard]] EquationPath::const_iterator end() const;

		[[nodiscard]] size_t depth() const;

		[[nodiscard]] Expression getExp() const;

		[[nodiscard]] const EquationPath& getEqPath() const;

		[[nodiscard]] bool isOnEquationLeftHand() const;

		[[nodiscard]] Expression reach(Expression exp) const;

		private:
		EquationPath path;
		std::shared_ptr<Expression> exp;
	};
}