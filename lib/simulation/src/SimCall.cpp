#include "modelica/simulation/SimCall.hpp"

#include <algorithm>

#include "modelica/simulation/SimExp.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

bool SimCall::operator==(const SimCall& other) const
{
	if (name != other.name)
		return false;

	if (args.size() != other.args.size())
		return false;

	return std::equal(
			begin(args),
			end(args),
			begin(other.args),
			[](const auto& left, const auto& right) { return *left == *right; });
}

SimCall::SimCall(string name, initializer_list<SimExp> arguments, SimType type)
		: name(move(name)), type(std::move(type))
{
	for (const auto& arg : arguments)
		args.push_back(std::make_unique<SimExp>(move(arg)));
}

SimCall::SimCall(const SimCall& other): name(other.name), type(other.type)
{
	for (const auto& arg : other.args)
		args.push_back(std::make_unique<SimExp>(*arg));
}

SimCall& SimCall::operator=(const SimCall& other)
{
	if (this == &other)
		return *this;
	name = other.name;

	args.clear();
	for (const auto& arg : other.args)
		args.push_back(std::make_unique<SimExp>(*arg));
	return *this;
}

void SimCall::dump(raw_ostream& OS) const
{
	OS << "call ";
	OS << name;
	OS << '(';
	for (size_t a = 0; a < argsSize(); a++)
	{
		at(a).dump(OS);

		if (a != argsSize() - 1)
			OS << ',';
	}
	OS << ')';
}
