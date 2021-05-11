#include <modelica/frontend/AST.h>

using namespace modelica;
using namespace frontend;

Member::Member(
		SourcePosition location,
		llvm::StringRef name,
		Type tp,
		TypePrefix typePrefix,
		llvm::Optional<std::unique_ptr<Expression>> initializer,
		bool isPublic,
		llvm::Optional<std::unique_ptr<Expression>> startOverload)
		: ASTNode(ASTNodeKind::MEMBER, std::move(location)),
			name(name.str()),
			type(std::move(tp)),
			typePrefix(typePrefix),
			initializer(std::move(initializer)),
			isPublicMember(isPublic),
			startOverload(std::move(startOverload))
{
}

Member::Member(const Member& other)
		: ASTNode(static_cast<const ASTNode&>(other)),
			name(other.name),
			type(other.type),
			typePrefix(other.typePrefix),
			initializer(other.initializer),
			isPublicMember(other.isPublicMember),
			startOverload(other.startOverload)
{
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

namespace modelica::frontend
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

void Member::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "member: {name: " << name << ", type: ";
	type.dump(os);
	os << "}\n";

	if (hasInitializer())
	{
		os.indent(indents + 1);
		os << "initializer:\n";
		initializer->get()->dump(os, indents + 2);
	}

	if (hasStartOverload())
	{
		os.indent(indents + 1);
		os << "start overload:\n";
		startOverload->get()->dump(os, indents + 2);
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
