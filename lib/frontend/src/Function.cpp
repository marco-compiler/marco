#include <algorithm>
#include <modelica/frontend/ClassContainer.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

template<typename T> using Container = Function::Container<T>;

Function::Function(
		SourcePosition location,
		string name,
		bool pure,
		ArrayRef<Member> members,
		ArrayRef<Algorithm> algorithms)
		: location(move(location)),
			name(move(name)),
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

SourcePosition Function::getSourcePosition() const
{
	return location;
}

string Function::getName() const { return name; }

Container<Member>& Function::getMembers() { return members; }

const Container<Member>& Function::getMembers() const { return members; }

void Function::addMember(Member member)
{
	members.emplace_back(move(member));
}

Container<Algorithm>& Function::getAlgorithms() { return algorithms; }

const Container<Algorithm>& Function::getAlgorithms() const
{
	return algorithms;
}

modelica::Type& Function::getType() { return type; }

const Type& Function::getType() const { return type; }

void Function::setType(Type t) { type = move(t); }
