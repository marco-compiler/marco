#include "modelica/frontend/ReferenceAccess.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

ReferenceAccess::ReferenceAccess(string name, bool globalLookup)
		: referencedName(move(name)), globalLookup(globalLookup)
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
	os << "reference access " << (globalLookup ? "." : "") << referencedName;
}

string& ReferenceAccess::getName() { return referencedName; }

const string& ReferenceAccess::getName() const { return referencedName; }

bool ReferenceAccess::hasGlobalLookup() const { return globalLookup; }
