#include <modelica/frontend/ClassContainer.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Class::Class(
		string name,
		ArrayRef<Member> members,
		ArrayRef<Equation> equations,
		ArrayRef<ForEquation> forEquations,
		ArrayRef<Algorithm> algorithms,
		ArrayRef<ClassContainer> innerClasses)
		: name(move(name)),
			members(members.begin(), members.end()),
			equations(equations.begin(), equations.end()),
			forEquations(forEquations.begin(), forEquations.end()),
			algorithms(algorithms.begin(), algorithms.end())
{
	assert(!this->name.empty());

	for (const auto& cls : innerClasses)
		this->innerClasses.emplace_back(std::make_unique<ClassContainer>(cls));
}

Class::Class(const Class& other)
		: name(other.name),
			members(other.members),
			equations(other.equations),
			forEquations(other.forEquations),
			algorithms(other.algorithms)
{
	innerClasses.clear();

	for (const auto& cls : other.innerClasses)
		innerClasses.push_back(std::make_unique<ClassContainer>(*cls));
}

Class& Class::operator=(const Class& other)
{
	if (this == &other)
		return *this;

	name = other.name;
	members = other.members;
	equations = other.equations;
	forEquations = other.forEquations,
			algorithms = other.algorithms;

	innerClasses.clear();

	for (const auto& cls : other.innerClasses)
		innerClasses.push_back(std::make_unique<ClassContainer>(*cls));

	return *this;
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

string Class::getName() const { return name; }

SmallVectorImpl<Member>& Class::getMembers() { return members; }

const SmallVectorImpl<Member>& Class::getMembers() const { return members; }

void Class::addMember(Member member)
{
	return members.push_back(move(member));
}

SmallVectorImpl<Equation>& Class::getEquations() { return equations; }

const SmallVectorImpl<Equation>& Class::getEquations() const
{
	return equations;
}

SmallVectorImpl<ForEquation>& Class::getForEquations() { return forEquations; }

const SmallVectorImpl<ForEquation>& Class::getForEquations() const
{
	return forEquations;
}

SmallVectorImpl<Algorithm>& Class::getAlgorithms() { return algorithms; }

const SmallVectorImpl<Algorithm>& Class::getAlgorithms() const
{
	return algorithms;
}

SmallVectorImpl<UniqueClass>& Class::getInnerClasses() { return innerClasses; }

const SmallVectorImpl<UniqueClass>& Class::getInnerClasses() const { return innerClasses; }

const Type& Function::getType() const { return type; }

void Function::setType(Type t) { type = move(t); }
