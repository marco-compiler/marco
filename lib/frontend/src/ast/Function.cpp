#include <algorithm>
#include <modelica/frontend/AST.h>
#include <variant>

using namespace modelica::frontend;

Function::Function(SourcePosition location,
									 bool pure,
									 llvm::StringRef name,
									 llvm::Optional<std::unique_ptr<Annotation>> annotation)
		: ASTNode(std::move(location)),
			pure(pure),
			name(name.str()),
			annotation(annotation.hasValue() ? llvm::Optional(annotation.getValue()->clone()) : llvm::None)
{
	if (annotation.hasValue())
		this->annotation = annotation.getValue()->clone();
	else
		this->annotation = llvm::None;
}

Function::Function(const Function& other)
		: ASTNode(other),
			pure(other.pure),
			name(other.name)
{
	if (other.annotation.hasValue())
		annotation = other.annotation.getValue()->clone();
	else
		annotation = llvm::None;
}

Function::Function(Function&& other) = default;

Function::~Function() = default;

Function& Function::operator=(const Function& other)
{
	if (this != &other)
	{
		static_cast<ASTNode&>(*this) = static_cast<const ASTNode&>(other);

		this->pure = other.pure;
		this->name = other.name;
		this->annotation = other.annotation.hasValue() ? llvm::Optional(other.annotation.getValue()->clone()) : llvm::None;
	}

	return *this;
}

Function& Function::operator=(Function&& other) = default;

namespace modelica::frontend
{
	void swap(Function& first, Function& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.pure, second.pure);
		swap(first.name, second.name);
		swap(first.annotation, second.annotation);
	}
}

llvm::StringRef Function::getName() const
{
	return name;
}

bool Function::isPure() const
{
	return pure;
}

bool Function::hasAnnotation() const
{
	return annotation.hasValue();
}

Annotation* Function::getAnnotation()
{
	assert(annotation.hasValue());
	return annotation.getValue().get();
}

const Annotation* Function::getAnnotation() const
{
	assert(annotation.hasValue());
	return annotation.getValue().get();
}

DerFunction::DerFunction(SourcePosition location,
												 bool pure,
												 llvm::StringRef name,
												 llvm::Optional<std::unique_ptr<Annotation>> annotation,
												 llvm::StringRef derivedFunction,
												 llvm::StringRef arg)
		: Function(std::move(location), pure, name, std::move(annotation)),
			derivedFunction(derivedFunction.str()),
			arg(arg.str())
{
}

DerFunction::DerFunction(const DerFunction& other)
		: Function(other),
			derivedFunction(other.derivedFunction),
			arg(other.arg)
{
}

DerFunction::DerFunction(DerFunction&& other) = default;

DerFunction::~DerFunction() = default;

DerFunction& DerFunction::operator=(const DerFunction& other)
{
	DerFunction result(other);
	swap(*this, result);
	return *this;
}

DerFunction& DerFunction::operator=(DerFunction&& other) = default;

namespace modelica::frontend
{
	void swap(DerFunction& first, DerFunction& second)
	{
		swap(static_cast<Function&>(first), static_cast<Function&>(second));

		using std::swap;
		swap(first.derivedFunction, second.derivedFunction);
		swap(first.arg, second.arg);
	}
}

void DerFunction::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "function " << getName() << ": der(" << getDerivedFunction() << ", " << getArg() << "\n";
}

llvm::StringRef DerFunction::getDerivedFunction() const
{
	return derivedFunction;
}

llvm::StringRef DerFunction::getArg() const
{
	return arg;
}

StandardFunction::StandardFunction(SourcePosition location,
																	 bool pure,
																	 llvm::StringRef name,
																	 llvm::Optional<std::unique_ptr<Annotation>> annotation,
																	 llvm::ArrayRef<std::unique_ptr<Member>> members,
																	 llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms)
		: Function(std::move(location), pure, name, std::move(annotation)),
			type(Type::unknown())
{
	for (const auto& member : members)
		this->members.push_back(member->clone());

	for (const auto& algorithm : algorithms)
		this->algorithms.push_back(algorithm->clone());
}

StandardFunction::StandardFunction(const StandardFunction& other)
		: Function(other),
			type(other.type)
{
	for (const auto& member : other.members)
		this->members.push_back(member->clone());

	for (const auto& algorithm : other.algorithms)
		this->algorithms.push_back(algorithm->clone());
}

StandardFunction::StandardFunction(StandardFunction&& other) = default;

StandardFunction::~StandardFunction() = default;

StandardFunction& StandardFunction::operator=(const StandardFunction& other)
{
	StandardFunction result(other);
	swap(*this, result);
	return *this;
}

StandardFunction& StandardFunction::operator=(StandardFunction&& other) = default;

namespace modelica::frontend
{
	void swap(StandardFunction& first, StandardFunction& second)
	{
		swap(static_cast<Function&>(first), static_cast<Function&>(second));

		impl::swap(first.members, second.members);
		impl::swap(first.algorithms, second.algorithms);
		std::swap(first.type, second.type);
	}
}

void StandardFunction::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "function " << getName() << "\n";

	for (const auto& member : members)
		member->print(os, indents + 1);

	for (const auto& algorithm : getAlgorithms())
		algorithm->print(os, indents + 1);
}

Member* StandardFunction::operator[](llvm::StringRef name)
{
	for (auto& member : members)
		if (member->getName() == name)
			return member.get();

	assert(false && "Not found");
}

const Member* StandardFunction::operator[](llvm::StringRef name) const
{
	for (const auto& member : members)
		if (member->getName() == name)
			return member.get();

	assert(false && "Not found");
}

llvm::MutableArrayRef<std::unique_ptr<Member>> StandardFunction::getMembers()
{
	return members;
}

llvm::ArrayRef<std::unique_ptr<Member>> StandardFunction::getMembers() const
{
	return members;
}

StandardFunction::Container<Member*> StandardFunction::getArgs() const
{
	Container<Member*> result;

	for (const auto& member : members)
		if (member->isInput())
			result.push_back(member.get());

	return result;
}

StandardFunction::Container<Member*> StandardFunction::getResults() const
{
	Container<Member*> result;

	for (const auto& member : members)
		if (member->isOutput())
			result.push_back(member.get());

	return result;
}

StandardFunction::Container<Member*> StandardFunction::getProtectedMembers() const
{
	Container<Member*> result;

	for (const auto& member : members)
		if (!member->isInput() && !member->isOutput())
			result.push_back(member.get());

	return result;
}

void StandardFunction::addMember(std::unique_ptr<Member> member)
{
	this->members.push_back(std::move(member));
}

llvm::MutableArrayRef<std::unique_ptr<Algorithm>> StandardFunction::getAlgorithms()
{
	return algorithms;
}

llvm::ArrayRef<std::unique_ptr<Algorithm>> StandardFunction::getAlgorithms() const
{
	return algorithms;
}

Type& StandardFunction::getType()
{
	return type;
}

const Type& StandardFunction::getType() const
{
	return type;
}

void StandardFunction::setType(Type newType)
{
	this->type = std::move(newType);
}

bool InverseFunctionAnnotation::isInvertible(llvm::StringRef arg) const
{
	return map.find(arg) != map.end();
}

llvm::StringRef InverseFunctionAnnotation::getInverseFunction(llvm::StringRef invertibleArg) const
{
	assert(isInvertible(invertibleArg));
	return map.find(invertibleArg)->second.first;
}

llvm::ArrayRef<std::string> InverseFunctionAnnotation::getInverseArgs(llvm::StringRef invertibleArg) const
{
	assert(isInvertible(invertibleArg));
	return map.find(invertibleArg)->second.second;
}

void InverseFunctionAnnotation::addInverse(llvm::StringRef invertedArg, llvm::StringRef inverseFunctionName, llvm::ArrayRef<std::string> args)
{
	assert(map.find(invertedArg) == map.end());
	Container<std::string> c(args.begin(), args.end());
	map[invertedArg] = std::make_pair(inverseFunctionName.str(), c);
}
