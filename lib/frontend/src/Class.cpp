#include <modelica/frontend/ClassContainer.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Class::Class(
		SourcePosition location,
		string name,
		ArrayRef<Member> members,
		ArrayRef<Equation> equations,
		ArrayRef<ForEquation> forEquations,
		ArrayRef<Algorithm> algorithms,
		ArrayRef<ClassContainer> innerClasses)
		: location(move(location)),
			name(move(name)),
			members(members.begin(), members.end()),
			equations(equations.begin(), equations.end()),
			forEquations(forEquations.begin(), forEquations.end()),
			algorithms(algorithms.begin(), algorithms.end())
{
	assert(!this->name.empty());

	for (const auto& cls : innerClasses)
		this->innerClasses.emplace_back(std::make_shared<ClassContainer>(cls));
}

void Class::dump() const { dump(outs(), 0); }

void Class::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "model " << name << "\n";

	for (const auto& member : members)
		member.dump(os, indents + 1);

	for (const auto& equation : equations)
		equation.dump(os, indents + 1);

	for (const auto& equation : forEquations)
		equation.dump(os, indents + 1);

	for (const auto& algorithm : algorithms)
		algorithm.dump(os, indents + 1);

	for (const auto& cls : innerClasses)
		cls->dump(os, indents + 1);
}

SourcePosition Class::getLocation() const
{
	return location;
}

string Class::getName() const { return name; }

Class::Container<Member>& Class::getMembers() { return members; }

const Class::Container<Member>& Class::getMembers() const { return members; }

void Class::addMember(Member member)
{
	return members.push_back(move(member));
}

Class::Container<Equation>& Class::getEquations() { return equations; }

const Class::Container<Equation>& Class::getEquations() const
{
	return equations;
}

Class::Container<ForEquation>& Class::getForEquations() { return forEquations; }

const Class::Container<ForEquation>& Class::getForEquations() const
{
	return forEquations;
}

Class::Container<Algorithm>& Class::getAlgorithms() { return algorithms; }

const Class::Container<Algorithm>& Class::getAlgorithms() const
{
	return algorithms;
}

Class::Container<Class::ClassPtr>& Class::getInnerClasses() { return innerClasses; }

const Class::Container<Class::ClassPtr>& Class::getInnerClasses() const { return innerClasses; }
