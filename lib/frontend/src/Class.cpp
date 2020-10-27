#include <modelica/frontend/Class.hpp>

using namespace std;
using namespace llvm;
using namespace modelica;

Class::Class(
		ClassType type,
		string name,
		ArrayRef<Member> members,
		ArrayRef<Equation> equations,
		ArrayRef<ForEquation> forEquations)
		: type(move(type)),
			name(move(name)),
			members(iterator_range<ArrayRef<Member>::iterator>(move(members))),
			equations(iterator_range<ArrayRef<Equation>::iterator>(move(equations))),
			forEquations(
					iterator_range<ArrayRef<ForEquation>::iterator>(move(forEquations)))
{
}

void Class::dump() const { dump(outs(), 0); }

void Class::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "class " << name << "\n";

	for (const auto& member : members)
		member.dump(os, indents + 1);

	for (const auto& equation : equations)
		equation.dump(os, indents + 1);

	for (const auto& equation : forEquations)
		equation.dump(os, indents + 1);

	for (const auto& algorithm : algorithms)
		algorithm.dump(os, indents + 1);
}

string& Class::getName() { return name; }

SmallVectorImpl<Member>& Class::getMembers() { return members; }

SmallVectorImpl<Equation>& Class::getEquations() { return equations; }

SmallVectorImpl<ForEquation>& Class::getForEquations() { return forEquations; }

SmallVectorImpl<Algorithm>& Class::getAlgorithms() { return algorithms; }

SmallVectorImpl<Func>& Class::getFunctions() { return functions; }

const string& Class::getName() const { return name; }

const SmallVectorImpl<Member>& Class::getMembers() const { return members; }

size_t Class::membersCount() const { return members.size(); }

const SmallVectorImpl<Equation>& Class::getEquations() const
{
	return equations;
}

const SmallVectorImpl<ForEquation>& Class::getForEquations() const
{
	return forEquations;
}

const SmallVectorImpl<Algorithm>& Class::getAlgorithms() const
{
	return algorithms;
}

const SmallVectorImpl<Func>& Class::getFunctions() const { return functions; }

void Class::addMember(Member newMember)
{
	return members.push_back(std::move(newMember));
}

void Class::eraseMember(size_t memberIndex)
{
	assert(memberIndex < members.size());
	members.erase(members.begin() + memberIndex);
}

void Class::addEquation(Equation equation)
{
	return equations.push_back(move(equation));
}

void Class::addForEquation(ForEquation equation)
{
	return forEquations.push_back(move(equation));
}

void Class::addAlgorithm(Algorithm algorithm)
{
	algorithms.push_back(move(algorithm));
}

void Class::addFunction(Class function)
{
	return functions.push_back(make_unique<Class>(move(function)));
}
