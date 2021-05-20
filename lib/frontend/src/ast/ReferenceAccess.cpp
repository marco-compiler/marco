#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

ReferenceAccess::ReferenceAccess(SourceRange location,
																 Type type,
																 llvm::StringRef name,
																 bool globalLookup,
																 bool dummy)
		: ASTNode(std::move(location)),
			type(std::move(type)),
			name(name.str()),
			globalLookup(globalLookup),
			dummyVar(dummy)
{
}

ReferenceAccess::ReferenceAccess(const ReferenceAccess& other)
		: ASTNode(other),
			type(other.type),
			name(other.name),
			globalLookup(other.globalLookup),
			dummyVar(other.dummyVar)
{
}

ReferenceAccess::ReferenceAccess(ReferenceAccess&& other) = default;

ReferenceAccess::~ReferenceAccess() = default;

ReferenceAccess& ReferenceAccess::operator=(const ReferenceAccess& other)
{
	ReferenceAccess result(other);
	swap(*this, result);
	return *this;
}

ReferenceAccess& ReferenceAccess::operator=(ReferenceAccess&& other) = default;

namespace modelica::frontend
{
	void swap(ReferenceAccess& first, ReferenceAccess& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.type, second.type);
		swap(first.name, second.name);
		swap(first.globalLookup, second.globalLookup);
		swap(first.dummyVar, second.dummyVar);
	}
}

void ReferenceAccess::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "reference access: " << (hasGlobalLookup() ? "." : "") << name << "\n";
}

bool ReferenceAccess::isLValue() const
{
	return true;
}

bool ReferenceAccess::operator==(const ReferenceAccess& other) const
{
	return globalLookup == other.globalLookup &&
				 name == other.name &&
				 dummyVar == other.dummyVar;
}

bool ReferenceAccess::operator!=(const ReferenceAccess& other) const
{
	return !(*this == other);
}

Type& ReferenceAccess::getType()
{
	return type;
}

const Type& ReferenceAccess::getType() const
{
	return type;
}

void ReferenceAccess::setType(Type tp)
{
	type = std::move(tp);
}

llvm::StringRef ReferenceAccess::getName() const
{
	return name;
}

void ReferenceAccess::setName(llvm::StringRef newName)
{
	this->name = newName.str();
}

bool ReferenceAccess::hasGlobalLookup() const
{
	return globalLookup;
}

bool ReferenceAccess::isDummy() const
{
	return dummyVar;
}

std::unique_ptr<Expression> ReferenceAccess::dummy(SourceRange location, Type type)
{
	return Expression::reference(location, type, "", false, true);
}

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const ReferenceAccess& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const ReferenceAccess& obj)
	{
		return obj.getName().str();
	}
}
