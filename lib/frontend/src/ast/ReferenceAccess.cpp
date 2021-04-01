#include <modelica/frontend/AST.h>

using namespace modelica;

ReferenceAccess::ReferenceAccess(SourcePosition location,
																 std::string name,
																 bool globalLookup,
																 bool dummy)
		: location(std::move(location)),
			referencedName(std::move(name)),
			globalLookup(globalLookup),
			dummyVariable(dummy)
{
}

bool ReferenceAccess::operator==(const ReferenceAccess& other) const
{
	return globalLookup == other.globalLookup &&
				 referencedName == other.referencedName;
}

bool ReferenceAccess::operator!=(const ReferenceAccess& other) const
{
	return !(*this == other);
}

void ReferenceAccess::dump() const { dump(llvm::outs(), 0); }

void ReferenceAccess::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "reference access: " << (globalLookup ? "." : "") << referencedName
		 << "\n";
}

SourcePosition ReferenceAccess::getLocation() const
{
	return location;
}

std::string& ReferenceAccess::getName() { return referencedName; }

const std::string& ReferenceAccess::getName() const { return referencedName; }

void ReferenceAccess::setName(std::string name) { referencedName = name; }

bool ReferenceAccess::hasGlobalLookup() const { return globalLookup; }

bool ReferenceAccess::isDummy() const { return dummyVariable; }

ReferenceAccess ReferenceAccess::dummy(SourcePosition location)
{
	return ReferenceAccess(location, "", false, true);
}

llvm::raw_ostream& modelica::operator<<(llvm::raw_ostream& stream, const ReferenceAccess& obj)
{
	return stream << toString(obj);
}

std::string modelica::toString(const ReferenceAccess& obj)
{
	return obj.getName();
}
