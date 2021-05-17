#include "modelica/model/ModMatchers.hpp"

#include "modelica/model/ModExpPath.hpp"

using namespace llvm;
using namespace std;
using namespace modelica;

void ReferenceMatcher::visit(const ModEquation& equation, bool ingnoreMatched)
{
	assert(!ingnoreMatched || equation.isMatched());
  /* riempi vars */
	visit(equation.getLeft(), true, 0);
	visit(equation.getRight(), false, 0);

  /* per lo scheduling non vogliamo considerare le sottoexpr matchate perché
   * altrimenti lo scheduling si confonde */
	if (!ingnoreMatched)
		return;

  /* espressione che "computa una variabile dentro un array"
   * nota: allo stadio in cui questo codice viene usato i matching parziali di
   * un'equazione vettoriale sono stati già gestiti spezzando l'equazione
   * vettoriale in più equazioni vettoriali/scalari che sono matchate totalmente */
	const auto* match = &equation.getMatchedExp();

	vars.erase(
			remove_if(
					vars,
					[match](const ModExpPath& path) { return &path.getExp() == match; }),
			vars.end());
}

void ReferenceMatcher::visit(const ModExp& exp, bool isLeft, size_t index)
{
	if (exp.isReferenceAccess() /* variabile o espressione con accessi a variabile vettore */)
	{
		vars.emplace_back(exp, currentPath, isLeft);
		return;
	}

	for (auto index : irange(exp.childCount()))
	{
		currentPath.push_back(index);
		auto g = makeGuard(std::bind(&ReferenceMatcher::removeBack, this));
		visit(exp.getChild(index), isLeft, index);
	}
}
