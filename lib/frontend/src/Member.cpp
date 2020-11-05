#include <modelica/frontend/Member.hpp>

using namespace llvm;
using namespace std;
using namespace modelica;

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

void Member::dump(llvm::raw_ostream& OS, size_t indents) const
{
	OS.indent(indents);
	OS << "member " << name << " type : ";
	type.dump(OS);
	OS << (isParameter() ? "param" : "");
	OS << "\n";

	if (hasInitializer())
	{
		OS.indent(indents);
		OS << "initializer: \n";
		initializer->dump(OS, indents + 1);
		OS << "\n";
	}

	if (hasStartOverload())
	{
		OS.indent(indents);
		OS << "start overload: ";
		startOverload->dump(OS);
		OS << "\n";
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
