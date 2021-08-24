#include "marco/matching/Edge.hpp"

using namespace marco;
using namespace std;
using namespace llvm;

void Edge::dump(llvm::raw_ostream& OS) const
{
	OS << "EDGE: Eq_" << equation << " TO " << variable->getName();
	OS << "\n";
	OS << "\tForward Map: ";
	vectorAccess.dump(OS);
	OS << " -> Backward Map: ";
	invertedAccess.dump(OS);
	OS << "\n\tCurrent Flow: ";
	set.dump(OS);
	OS << "\n";
}

string Edge::toString() const
{
	string str;
	raw_string_ostream ss(str);
	dump(ss);
	ss.flush();
	return str;
}
