#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>

using namespace modelica::frontend;

Induction::Induction(SourcePosition location,
										 llvm::StringRef inductionVariable,
										 std::unique_ptr<Expression> begin,
										 std::unique_ptr<Expression> end)
		: ASTNodeCRTP<Induction>(ASTNodeKind::INDUCTION, std::move(location)),
			inductionVariable(inductionVariable.str()),
			begin(std::move(begin)),
			end(std::move(end)),
			inductionIndex(0)
{
}

Induction::Induction(const Induction& other)
		: ASTNodeCRTP<Induction>(static_cast<const ASTNodeCRTP<Induction>&>(other)),
			inductionVariable(other.inductionVariable),
			begin(other.begin->cloneExpression()),
			end(other.end->cloneExpression()),
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
		swap(static_cast<impl::ASTNodeCRTP<Induction>&>(first),
				 static_cast<impl::ASTNodeCRTP<Induction>&>(second));

		using std::swap;
		swap(first.inductionVariable, second.inductionVariable);
		swap(first.begin, second.begin);
		swap(first.end, second.end);
		swap(first.inductionIndex, second.inductionIndex);
	}
}

void Induction::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "induction var " << getName() << "\n";

	os.indent(indents);
	os << "from ";
	begin->dump(os, indents + 1);
	os << "\n";
	os.indent(indents);
	os << "to";
	end->dump(os, indents + 1);
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
