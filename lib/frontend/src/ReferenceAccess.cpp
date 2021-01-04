#include <modelica/frontend/ReferenceAccess.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

ReferenceAccess::ReferenceAccess(SourcePosition location, string name, bool globalLookup, bool dummy)
		: location(move(location)),
			referencedName(move(name)),
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

void ReferenceAccess::dump() const { dump(outs(), 0); }

void ReferenceAccess::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "reference access: " << (globalLookup ? "." : "") << referencedName
		 << "\n";
}

SourcePosition ReferenceAccess::getLocation() const
{
	return location;
}

string& ReferenceAccess::getName() { return referencedName; }

const string& ReferenceAccess::getName() const { return referencedName; }

void ReferenceAccess::setName(std::string name) { referencedName = name; }

bool ReferenceAccess::hasGlobalLookup() const { return globalLookup; }

bool ReferenceAccess::isDummy() const { return dummyVariable; }

ReferenceAccess ReferenceAccess::dummy(SourcePosition location)
{
	return ReferenceAccess(location, "", false, true);
}
