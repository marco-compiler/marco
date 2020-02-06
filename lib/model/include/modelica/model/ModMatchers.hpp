#pragma once
#include <functional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/utils/ScopeGuard.hpp"

namespace modelica
{
	class ReferenceMatcher
	{
		public:
		ReferenceMatcher() = default;
		ReferenceMatcher(const ModEquation& eq) { visit(eq); }

		void visit(const ModExp& exp, bool isLeft)
		{
			currentPath.push_back(&exp);
			auto g = makeGuard(std::bind(&ReferenceMatcher::removeBack, this));

			if (exp.isReferenceAccess())
			{
				vars.emplace_back(&exp);
				return;
			}
			for (const ModExp& child : exp)
				visit(child, isLeft);
		}

		void visit(const ModEquation& equation)
		{
			visit(equation.getLeft(), true);
			visit(equation.getRight(), false);
		}

		[[nodiscard]] auto begin() const { return vars.begin(); }
		[[nodiscard]] auto end() const { return vars.end(); }
		[[nodiscard]] auto begin() { return vars.begin(); }
		[[nodiscard]] auto end() { return vars.end(); }
		[[nodiscard]] size_t size() const { return vars.size(); }
		[[nodiscard]] const ModExp& at(size_t index) const { return *vars[index]; }
		[[nodiscard]] const ModExp& operator[](size_t index) const
		{
			return at(index);
		}

		private:
		void removeBack()
		{
			currentPath.erase(currentPath.end() - 1, currentPath.end());
		}
		llvm::SmallVector<const ModExp*, 3> currentPath;
		llvm::SmallVector<const ModExp*, 3> vars;
	};

};	// namespace modelica
