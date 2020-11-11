#include <modelica/frontend/Member.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Member::Member(
		string name,
		Type tp,
		TypePrefix typePrefix,
		Expression initializer,
		bool isPublic,
		optional<Expression> startOverload)
		: name(move(name)),
			type(move(tp)),
			typePrefix(typePrefix),
			initializer(move(initializer)),
			isPublicMember(isPublic),
			startOverload(move(startOverload))
{
}

Member::Member(
		string name,
		Type tp,
		TypePrefix typePrefix,
		bool isPublic,
		optional<Expression> startOverload)
		: name(move(name)),
			type(move(tp)),
			typePrefix(typePrefix),
			initializer(nullopt),
			isPublicMember(isPublic),
			startOverload(std::move(startOverload))
{
}

void Member::dump() const { dump(outs(), 0); }

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
		initializer->dump(os, indents + 2);
	}

	if (hasStartOverload())
	{
		os.indent(indents + 1);
		os << "start overload:\n";
		startOverload->dump(os, indents + 2);
	}
}

bool Member::operator==(const Member& other) const
{
	return name == other.name && type == other.type &&
				 initializer == other.initializer;
}

bool Member::operator!=(const Member& other) const { return !(*this == other); }

string& Member::getName() { return name; }

const string& Member::getName() const { return name; }

Type& Member::getType() { return type; }

const Type& Member::getType() const { return type; }

bool Member::hasInitializer() const { return initializer.has_value(); }

const Expression& Member::getInitializer() const
{
	assert(hasInitializer());
	return *initializer;
}

Expression& Member::getInitializer()
{
	assert(hasInitializer());
	return *initializer;
}

bool Member::hasStartOverload() const { return startOverload.has_value(); }

const Expression& Member::getStartOverload() const
{
	assert(hasStartOverload());
	return startOverload.value();
}

Expression& Member::getStartOverload()
{
	assert(hasStartOverload());
	return startOverload.value();
}

bool Member::isPublic() const { return isPublicMember; }

bool Member::isParameter() const { return typePrefix.isParameter(); }

bool Member::isInput() const { return typePrefix.isInput(); }

bool Member::isOutput() const { return typePrefix.isOutput(); }
