#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Model::Model(SourcePosition location,
						 llvm::StringRef name,
						 llvm::ArrayRef<std::unique_ptr<Member>> members,
						 llvm::ArrayRef<std::unique_ptr<Equation>> equations,
						 llvm::ArrayRef<std::unique_ptr<ForEquation>> forEquations,
						 llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms,
						 llvm::ArrayRef<std::unique_ptr<Class>> innerClasses)
		: ClassCRTP<Model>(ASTNodeKind::CLASS_MODEL, std::move(location), std::move(name))
{
	for (const auto& member : members)
		this->members.push_back(member->clone());

	for (const auto& equation : equations)
		this->equations.push_back(equation->clone());

	for (const auto& forEquation : forEquations)
		this->forEquations.push_back(forEquation->clone());

	for (const auto& algorithm : algorithms)
		this->algorithms.push_back(algorithm->clone());

	for (const auto& cls : innerClasses)
		this->innerClasses.push_back(cls->cloneClass());
}

Model::Model(const Model& other)
		: ClassCRTP<Model>(static_cast<ClassCRTP<Model>&>(*this))
{
	for (const auto& member : other.members)
		this->members.push_back(member->clone());

	for (const auto& equation : other.equations)
		this->equations.push_back(equation->clone());

	for (const auto& forEquation : other.forEquations)
		this->forEquations.push_back(forEquation->clone());

	for (const auto& algorithm : other.algorithms)
		this->algorithms.push_back(algorithm->clone());

	for (const auto& cls : other.innerClasses)
		this->innerClasses.push_back(cls->cloneClass());
}

Model::Model(Model&& other) = default;

Model::~Model() = default;

Model& Model::operator=(const Model& other)
{
	Model result(other);
	swap(*this, result);
	return *this;
}

Model& Model::operator=(Model&& other) = default;

namespace modelica::frontend
{
	void swap(Model& first, Model& second)
	{
		swap(static_cast<impl::ClassCRTP<Model>&>(first),
				 static_cast<impl::ClassCRTP<Model>&>(second));

		impl::swap(first.members, second.members);
		impl::swap(first.equations, second.equations);
		impl::swap(first.forEquations, second.forEquations);
		impl::swap(first.algorithms, second.algorithms);
		impl::swap(first.innerClasses, second.innerClasses);
	}
}

void Model::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "model " << getName() << "\n";

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

llvm::MutableArrayRef<std::unique_ptr<Member>> Model::getMembers()
{
	return members;
}

llvm::ArrayRef<std::unique_ptr<Member>> Model::getMembers() const
{
	return members;
}

void Model::addMember(Member* member)
{
	members.push_back(member->clone());
}

llvm::MutableArrayRef<std::unique_ptr<Equation>> Model::getEquations()
{
	return equations;
}

llvm::ArrayRef<std::unique_ptr<Equation>> Model::getEquations() const
{
	return equations;
}

llvm::MutableArrayRef<std::unique_ptr<ForEquation>> Model::getForEquations()
{
	return forEquations;
}

llvm::ArrayRef<std::unique_ptr<ForEquation>> Model::getForEquations() const
{
	return forEquations;
}

llvm::MutableArrayRef<std::unique_ptr<Algorithm>> Model::getAlgorithms()
{
	return algorithms;
}

llvm::ArrayRef<std::unique_ptr<Algorithm>> Model::getAlgorithms() const
{
	return algorithms;
}

llvm::MutableArrayRef<std::unique_ptr<Class>> Model::getInnerClasses()
{
	return innerClasses;
}

llvm::ArrayRef<std::unique_ptr<Class>> Model::getInnerClasses() const
{
	return innerClasses;
}
