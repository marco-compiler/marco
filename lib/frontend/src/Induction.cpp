#include <modelica/frontend/Induction.hpp>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Induction::Induction(string indVar, Expression begin, Expression end)
		: begin(move(begin)),
			end(move(end)),
			inductionIndex(0),
			inductionVar(move(indVar))
{
}

void Induction::dump() const { dump(outs(), 0); }

void Induction::dump(raw_ostream& os, size_t indents) const
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

const string& Induction::getName() const { return inductionVar; }

Expression& Induction::getBegin() { return begin; }

const Expression& Induction::getBegin() const { return begin; }

Expression& Induction::getEnd() { return end; }

const Expression& Induction::getEnd() const { return end; }

size_t Induction::getInductionIndex() const { return inductionIndex; }

void Induction::setInductionIndex(size_t index) { inductionIndex = index; }
