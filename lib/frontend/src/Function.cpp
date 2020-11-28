#include <modelica/frontend/ClassContainer.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Function::Function(
		string name,
		bool pure,
		ArrayRef<Member> members,
		ArrayRef<Algorithm> algorithms)
		: name(move(name)),
			pure(pure),
			members(members.begin(), members.end()),
			algorithms(algorithms.begin(), algorithms.end()),
			type(Type::unknown())
{
	assert(!this->name.empty());
}

void Function::dump() const { dump(outs(), 0); }

void Function::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "function " << name << "\n";

	for (const auto& member : members)
		member.dump(os, indents + 1);

	for (const auto& algorithm : algorithms)
		algorithm.dump(os, indents + 1);
}

string Function::getName() const { return name; }

SmallVectorImpl<Member>& Function::getMembers() { return members; }

const SmallVectorImpl<Member>& Function::getMembers() const { return members; }

void Function::addMember(Member member)
{
	members.emplace_back(move(member));
}

SmallVectorImpl<Algorithm>& Function::getAlgorithms() { return algorithms; }

const SmallVectorImpl<Algorithm>& Function::getAlgorithms() const
{
	return algorithms;
}

Type& Function::getType() { return type; }
