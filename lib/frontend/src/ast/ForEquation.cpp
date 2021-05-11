#include <memory>
#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>

using namespace modelica::frontend;

ForEquation::ForEquation(SourcePosition location,
												 llvm::ArrayRef<std::unique_ptr<Induction>> inductions,
												 std::unique_ptr<Equation> equation)
		: ASTNodeCRTP<ForEquation>(ASTNodeKind::FOR_EQUATION, std::move(location)),
			equation(std::move(equation))
{
	for (const auto& induction : inductions)
		this->inductions.push_back(induction->clone());
}

ForEquation::ForEquation(const ForEquation& other)
		: ASTNodeCRTP<ForEquation>(static_cast<const ASTNodeCRTP<ForEquation>&>(other)),
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

namespace modelica::frontend
{
	void swap(ForEquation& first, ForEquation& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<ForEquation>&>(first),
				 static_cast<impl::ASTNodeCRTP<ForEquation>&>(second));

		using std::swap;
		impl::swap(first.inductions, second.inductions);
		swap(first.equation, second.equation);
	}
}

void ForEquation::dump(llvm::raw_ostream& os, size_t indents) const
{
	os << "for equation\n";

	for (const auto& induction : getInductions())
	{
		induction->dump(os, indents + 1);
		os << "\n";
	}

	equation->dump(os, indents + 1);
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

Equation* ForEquation::getEquation() const
{
	return equation.get();
}
