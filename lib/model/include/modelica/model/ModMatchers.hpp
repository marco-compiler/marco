#pragma once
#include <functional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/utils/IRange.hpp"
#include "modelica/utils/ScopeGuard.hpp"

namespace modelica
{
  /*
   * data un'equazione produciamo il vettore di puntatori che puntano alle
   * sottoespressioni dell'albero che corrispondono a un accesso a
   * variabile
   *
   * esempio:
   *   der(u[j]) = ((-u[j]) + u[j-1])*10 - mu*u[j]*(u[j] - alpha)*(u[j] - 1);
   * vars generato:
   *   1. u[j] (quello nella der)
   *   2. u[j] (il secondo)
   *   3. u[j-1] (il terzo)
   *   4. mu (perché non sa che è una costante)
   *   5. u[j] (il quarto)
   *   6. u[j] (il quinto)
   *   7. u[j] (il sesto)
   */
	class ReferenceMatcher
	{
		public:
		ReferenceMatcher() = default;
		ReferenceMatcher(const ModEquation& eq) { visit(eq); }

		void visit(const ModExp& exp, bool isLeft, size_t index);

		void visit(const ModEquation& equation, bool ingnoreMatched = false);

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

};	// namespace modelica
