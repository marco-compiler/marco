#include "marco/model/ModMatchers.hpp"

#include "marco/model/ModExpPath.hpp"

using namespace llvm;
using namespace std;
using namespace marco;

void ReferenceMatcher::visit(
		const std::variant<ModEquation, ModBltBlock>& content, bool ignoreMatched)
{
	if (holds_alternative<ModEquation>(content))
	{
		const ModEquation& equation = get<ModEquation>(content);
		assert(!ignoreMatched || equation.isMatched());
		visit(equation.getLeft(), true, 0);
		visit(equation.getRight(), false, 0);

		if (!ignoreMatched)
			return;

		const ModExp* match = &equation.getMatchedExp();

		vars.erase(
				remove_if(
						vars,
						[match](const ModExpPath& path) {
							return &path.getExp() == match;
						}),
				vars.end());
	}
	else
	{
		for (ModEquation equation : get<ModBltBlock>(content).getEquations())
		{
			assert(!ignoreMatched || equation.isMatched());
			visit(equation.getLeft(), true, 0);
			visit(equation.getRight(), false, 0);

			if (!ignoreMatched)
				return;

			const ModExp* match = &equation.getMatchedExp();

			vars.erase(
					remove_if(
							vars,
							[match](const ModExpPath& path) {
								return &path.getExp() == match;
							}),
					vars.end());
		}
	}
}

void ReferenceMatcher::visit(const ModExp& exp, bool isLeft, size_t index)
{
	if (exp.isReferenceAccess())
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
