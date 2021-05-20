#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>

using namespace modelica::frontend;

Induction::Induction(SourceRange location,
										 llvm::StringRef inductionVariable,
										 std::unique_ptr<Expression> begin,
										 std::unique_ptr<Expression> end)
		: ASTNode(std::move(location)),
			inductionVariable(inductionVariable.str()),
			begin(std::move(begin)),
			end(std::move(end)),
			inductionIndex(0)
{
}

Induction::Induction(const Induction& other)
		: ASTNode(other),
			inductionVariable(other.inductionVariable),
			begin(other.begin->clone()),
			end(other.end->clone()),
			inductionIndex(other.inductionIndex)
{
}

Induction::Induction(Induction&& other) = default;

Induction::~Induction() = default;

Induction& Induction::operator=(const Induction& other)
{
	Induction result(other);
	swap(*this, result);
	return *this;
}

Induction& Induction::operator=(Induction&& other) = default;

namespace modelica::frontend
{
	void swap(Induction& first, Induction& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.inductionVariable, second.inductionVariable);
		swap(first.begin, second.begin);
		swap(first.end, second.end);
		swap(first.inductionIndex, second.inductionIndex);
	}
}

void Induction::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "induction var " << getName() << "\n";

	os.indent(indents);
	os << "from ";
	begin->print(os, indents + 1);
	os << "\n";
	os.indent(indents);
	os << "to";
	end->print(os, indents + 1);
}

llvm::StringRef Induction::getName() const
{
	return inductionVariable;
}

Expression* Induction::getBegin()
{
	return begin.get();
}

const Expression* Induction::getBegin() const
{
	return begin.get();
}

Expression* Induction::getEnd()
{
	return end.get();
}

const Expression* Induction::getEnd() const
{
	return end.get();
}

size_t Induction::getInductionIndex() const
{
	return inductionIndex;
}

void Induction::setInductionIndex(size_t index)
{
	this->inductionIndex = index;
}
