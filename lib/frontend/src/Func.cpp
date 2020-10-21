#include "modelica/frontend/Func.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

Func::Func(
		string name,
		ArrayRef<Member> input,
		ArrayRef<Member> output,
		ArrayRef<Member> members,
		ArrayRef<Algorithm> algorithms)
		: name(move(name)),
			input(iterator_range<ArrayRef<Member>::iterator>(move(input))),
			output(iterator_range<ArrayRef<Member>::iterator>(move(output))),
			members(iterator_range<ArrayRef<Member>::iterator>(move(members))),
			algorithms(
					iterator_range<ArrayRef<Algorithm>::iterator>(move(algorithms)))
{
}

void Func::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "function " << name << "\n";

	for (const auto& member : members)
		member.dump(os, indents + 1);

	for (const auto& algorithm : algorithms)
		algorithm.dump(os, indents + 1);
}

string& Func::getName() { return name; }

SmallVectorImpl<Member>& Func::getInputVariables() { return input; }

SmallVectorImpl<Member>& Func::getOutputVariables() { return output; }

SmallVectorImpl<Member>& Func::getMembers() { return members; }

SmallVectorImpl<Algorithm>& Func::getAlgorithms() { return algorithms; }

const string& Func::getName() const { return name; }

const SmallVectorImpl<Member>& Func::getInputVariables() const { return input; }

const SmallVectorImpl<Member>& Func::getOutputVariables() const
{
	return output;
}

const SmallVectorImpl<Member>& Func::getMembers() const { return members; }

const SmallVectorImpl<Algorithm>& Func::getAlgorithms() const
{
	return algorithms;
}
