#include <modelica/frontend/AST.h>

using namespace modelica;
using namespace frontend;

Class::Class(
		SourcePosition location,
		std::string name,
		llvm::ArrayRef<Member> members,
		llvm::ArrayRef<Equation> equations,
		llvm::ArrayRef<ForEquation> forEquations,
		llvm::ArrayRef<Algorithm> algorithms,
		llvm::ArrayRef<ClassContainer> innerClasses)
		: location(std::move(location)),
			name(std::move(name))
{
	assert(!this->name.empty());

	for (const auto& member : members)
		this->members.emplace_back(std::make_shared<Member>(member));

	for (const auto& equation : equations)
		this->equations.emplace_back(std::make_shared<Equation>(equation));

	for (const auto& forEquation : forEquations)
		this->forEquations.emplace_back(std::make_shared<ForEquation>(forEquation));

	for (const auto& algorithm : algorithms)
		this->algorithms.emplace_back(std::make_shared<Algorithm>(algorithm));

	for (const auto& cls : innerClasses)
		this->innerClasses.emplace_back(std::make_shared<ClassContainer>(cls));
}

void Class::dump() const { dump(llvm::outs(), 0); }

void Class::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "model " << name << "\n";

	for (const auto& member : members)
		member->dump(os, indents + 1);

	for (const auto& equation : equations)
		equation->dump(os, indents + 1);

	for (const auto& equation : forEquations)
		equation->dump(os, indents + 1);

	for (const auto& algorithm : algorithms)
		algorithm->dump(os, indents + 1);

	for (const auto& cls : innerClasses)
		cls->dump(os, indents + 1);
}

SourcePosition Class::getLocation() const
{
	return location;
}

const std::string& Class::getName() const { return name; }

Class::Container<Member>& Class::getMembers() { return members; }

const Class::Container<Member>& Class::getMembers() const { return members; }

void Class::addMember(Member member)
{
	members.emplace_back(std::make_shared<Member>(std::move(member)));
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

Class::Container<ClassContainer>& Class::getInnerClasses() { return innerClasses; }

const Class::Container<ClassContainer>& Class::getInnerClasses() const { return innerClasses; }
