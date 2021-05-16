#pragma once
#include <functional>
#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
<<<<<<< HEAD:lib/model/include/marco/model/ModMatchers.hpp
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModExpPath.hpp"
#include "marco/utils/IRange.hpp"
#include "marco/utils/ScopeGuard.hpp"
=======
#include "modelica/model/ModBltBlock.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/utils/IRange.hpp"
#include "modelica/utils/ScopeGuard.hpp"
>>>>>>> Started adding ModBltBlocks to Scheduling:lib/model/include/modelica/model/ModMatchers.hpp

namespace marco
{
	class ReferenceMatcher
	{
		public:
		ReferenceMatcher() = default;
		ReferenceMatcher(const std::variant<ModEquation, ModBltBlock>& content)
		{
			visit(content);
		}

		void visit(const ModExp& exp, bool isLeft, size_t index);

		void visit(
				const std::variant<ModEquation, ModBltBlock>& content,
				bool ignoreMatched = false);

		[[nodiscard]] auto begin() const { return vars.begin(); }
		[[nodiscard]] auto end() const { return vars.end(); }
		[[nodiscard]] auto begin() { return vars.begin(); }
		[[nodiscard]] auto end() { return vars.end(); }
		[[nodiscard]] size_t size() const { return vars.size(); }
		[[nodiscard]] const ModExpPath& at(size_t index) const
		{
			return vars[index];
		}
		[[nodiscard]] ModExpPath& at(size_t index) { return vars[index]; }
		[[nodiscard]] const ModExpPath& operator[](size_t index) const
		{
			return at(index);
		}
		[[nodiscard]] ModExpPath& operator[](size_t index) { return at(index); }
		[[nodiscard]] const ModExp& getExp(size_t index) const
		{
			return at(index).getExp();
		}

		private:
		void removeBack()
		{
			currentPath.erase(currentPath.end() - 1, currentPath.end());
		}
		llvm::SmallVector<size_t, 3> currentPath;
		llvm::SmallVector<ModExpPath, 3> vars;
	};

};	// namespace marco
