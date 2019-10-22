#include "modelica/model/ModCall.hpp"

#include <algorithm>

#include "modelica/model/ModExp.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

bool ModCall::operator==(const ModCall& other) const
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

ModCall::ModCall(string name, initializer_list<ModExp> arguments, ModType type)
		: name(move(name)), type(std::move(type))
{
	for (const auto& arg : arguments)
		args.push_back(std::make_unique<ModExp>(move(arg)));
}

ModCall::ModCall(const ModCall& other): name(other.name), type(other.type)
{
	for (const auto& arg : other.args)
		args.push_back(std::make_unique<ModExp>(*arg));
}

ModCall& ModCall::operator=(const ModCall& other)
{
	if (this == &other)
		return *this;
	name = other.name;

	args.clear();
	for (const auto& arg : other.args)
		args.push_back(std::make_unique<ModExp>(*arg));
	return *this;
}

void ModCall::dump(raw_ostream& OS) const
{
	OS << "call ";
	OS << name;
	OS << ' ';
	type.dump(OS);
	OS << '(';
	for (size_t a = 0; a < args.size(); a++)
		at(a).dump(OS);
	OS << ')';
}
