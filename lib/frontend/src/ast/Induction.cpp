#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>

using namespace modelica::frontend;

Induction::Induction(std::string indVar, Expression begin, Expression end)
		: begin(std::move(begin)),
			end(std::move(end)),
			inductionIndex(0),
			inductionVar(move(indVar))
{
}

void Induction::dump() const { dump(llvm::outs(), 0); }

void Induction::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "induction var " << inductionVar << "\n";

	os.indent(indents);
	os << "from ";
	begin.dump(os, indents + 1);
	os << "\n";
	os.indent(indents);
	os << "to";
	end.dump(os, indents + 1);
}

const std::string& Induction::getName() const { return inductionVar; }

Expression& Induction::getBegin() { return begin; }

const Expression& Induction::getBegin() const { return begin; }

Expression& Induction::getEnd() { return end; }

const Expression& Induction::getEnd() const { return end; }

size_t Induction::getInductionIndex() const { return inductionIndex; }

void Induction::setInductionIndex(size_t index) { inductionIndex = index; }
