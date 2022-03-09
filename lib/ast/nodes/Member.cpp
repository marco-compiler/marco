#include "marco/ast/AST.h"

using namespace marco;
using namespace marco::ast;

Member::Member(
		SourceRange location,
		llvm::StringRef name,
		Type tp,
		TypePrefix typePrefix,
		llvm::Optional<std::unique_ptr<Expression>> initializer,
		bool isPublic,
		llvm::Optional<std::unique_ptr<Expression>> startOverload)
		: ASTNode(std::move(location)),
			name(name.str()),
			type(std::move(tp)),
			typePrefix(std::move(typePrefix)),
			isPublicMember(isPublic)
{
	if (initializer.hasValue())
		this->initializer = initializer.getValue()->clone();
	else
		this->initializer = llvm::None;

	if (startOverload.hasValue())
		this->startOverload = startOverload.getValue()->clone();
	else
		this->startOverload = llvm::None;
}

Member::Member(const Member& other)
		: ASTNode(other),
			name(other.name),
			type(other.type),
			typePrefix(other.typePrefix),
			isPublicMember(other.isPublicMember)
{
	if (other.initializer.hasValue())
		initializer = other.initializer.getValue()->clone();
	else
		initializer = llvm::None;

	if (other.startOverload.hasValue())
		startOverload = other.startOverload.getValue()->clone();
	else
		startOverload = llvm::None;
}

Member::Member(Member&& other) = default;

Member::~Member() = default;

Member& Member::operator=(const Member& other)
{
	Member result(other);
	swap(*this, result);
	return *this;
}

Member& Member::operator=(Member&& other) = default;

namespace marco::ast
{
	void swap(Member& first, Member& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.name, second.name);
		swap(first.type, second.type);
		swap(first.typePrefix, second.typePrefix);
		swap(first.initializer, second.initializer);
		swap(first.isPublicMember, second.isPublicMember);
		swap(first.startOverload, second.startOverload);
	}
}

void Member::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "member: {name: " << name << ", type: ";
	type.print(os);
	os << "}\n";

	if (hasInitializer())
	{
		os.indent(indents + 1);
		os << "initializer:\n";
		initializer.getValue()->print(os, indents + 2);
	}

	if (hasStartOverload())
	{
		os.indent(indents + 1);
		os << "start overload:\n";
		startOverload.getValue()->print(os, indents + 2);
	}
}

bool Member::operator==(const Member& other) const
{
	return name == other.name && type == other.type &&
				 initializer == other.initializer;
}

bool Member::operator!=(const Member& other) const
{
	return !(*this == other);
}

llvm::StringRef Member::getName() const
{
	return name;
}

Type& Member::getType() { return type; }

const Type& Member::getType() const { return type; }

bool Member::hasInitializer() const
{
	return initializer.hasValue();
}

Expression* Member::getInitializer()
{
	assert(hasInitializer());
	return initializer->get();
}

const Expression* Member::getInitializer() const
{
	assert(hasInitializer());
	return initializer->get();
}

bool Member::hasStartOverload() const
{
	return startOverload.hasValue();
}

Expression* Member::getStartOverload()
{
	assert(hasStartOverload());
	return startOverload->get();
}

const Expression* Member::getStartOverload() const
{
	assert(hasStartOverload());
	return startOverload->get();
}

bool Member::isPublic() const
{
	return isPublicMember;
}

bool Member::isParameter() const
{
	return typePrefix.isParameter();
}

bool Member::isInput() const
{
	return typePrefix.isInput();
}

bool Member::isOutput() const
{
	return typePrefix.isOutput();
}
