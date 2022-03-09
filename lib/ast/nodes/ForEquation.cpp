#include "marco/ast/AST.h"
#include "marco/utils/IRange.h"
#include <memory>

using namespace marco::ast;

ForEquation::ForEquation(SourceRange location,
												 llvm::ArrayRef<std::unique_ptr<Induction>> inductions,
												 std::unique_ptr<Equation> equation)
		: ASTNode(std::move(location)),
			equation(std::move(equation))
{
	for (const auto& induction : inductions)
		this->inductions.push_back(induction->clone());
}

ForEquation::ForEquation(const ForEquation& other)
		: ASTNode(other),
			equation(other.equation->clone())
{
	for (const auto& induction : other.inductions)
		this->inductions.push_back(induction->clone());
}

ForEquation::ForEquation(ForEquation&& other) = default;

ForEquation::~ForEquation() = default;

ForEquation& ForEquation::operator=(const ForEquation& other)
{
	ForEquation result(other);
	swap(*this, result);
	return *this;
}

ForEquation& ForEquation::operator=(ForEquation&& other) = default;

namespace marco::ast
{
	void swap(ForEquation& first, ForEquation& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		impl::swap(first.inductions, second.inductions);
		swap(first.equation, second.equation);
	}
}

void ForEquation::print(llvm::raw_ostream& os, size_t indents) const
{
	os << "for equation\n";

	for (const auto& induction : getInductions())
	{
		induction->print(os, indents + 1);
		os << "\n";
	}

	equation->print(os, indents + 1);
}

llvm::MutableArrayRef<std::unique_ptr<Induction>> ForEquation::getInductions()
{
	return inductions;
}

llvm::ArrayRef<std::unique_ptr<Induction>> ForEquation::getInductions() const
{
	return inductions;
}

size_t ForEquation::inductionsCount() const
{
	return inductions.size();
}

void ForEquation::addOuterInduction(std::unique_ptr<Induction> induction)
{
	inductions.insert(inductions.begin(), std::move(induction));
}

Equation* ForEquation::getEquation() const
{
	return equation.get();
}
