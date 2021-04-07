#pragma once

#include <llvm/ADT/SmallVector.h>

#include "Expression.h"
#include "Model.h"
#include "Path.h"

namespace modelica::codegen::model
{
	class ReferenceMatcher
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<ExpressionPath>::iterator;
		using const_iterator = Container<ExpressionPath>::const_iterator;

		ReferenceMatcher();
		ReferenceMatcher(const Equation& eq);

		[[nodiscard]] ExpressionPath& operator[](size_t index);
		[[nodiscard]] const ExpressionPath& operator[](size_t index) const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] ExpressionPath& at(size_t index);
		[[nodiscard]] const ExpressionPath& at(size_t index) const;

		[[nodiscard]] const Expression& getExp(size_t index) const;

		void visit(const Expression& exp, bool isLeft, size_t index);
		void visit(const Equation& equation, bool ignoreMatched = false);

		private:
		void removeBack();

		Container<size_t> currentPath;
		Container<ExpressionPath> vars;
	};

};	// namespace modelica
