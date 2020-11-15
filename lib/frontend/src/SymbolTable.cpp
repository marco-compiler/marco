#include <modelica/frontend/SymbolTable.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Symbol::Symbol(Class& clas): content(&clas) {}
Symbol::Symbol(Member& mem): content(&mem) {}
Symbol::Symbol(Induction& mem): content(&mem) {}

SymbolTable::SymbolTable() {}

SymbolTable::SymbolTable(const SymbolTable* parent): parentTable(parent) {}

SymbolTable::SymbolTable(Class& cls, const SymbolTable* parent)
		: parentTable(parent)
{
	addSymbol(cls);

	for (auto& member : cls.getMembers())
		addSymbol(member);

	for (auto& function : cls.getFunctions())
		addSymbol(*function);
}

bool SymbolTable::hasSymbol(llvm::StringRef name) const
{
	if (table.find(name) != table.end())
		return true;

	if (parentTable == nullptr)
		return false;

	return parentTable->hasSymbol(name);
}
