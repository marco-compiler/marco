#include <modelica/frontend/SymbolTable.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Symbol::Symbol(Function& function): content(&function) {}

Symbol::Symbol(Class& model): content(&model) {}

Symbol::Symbol(Member& mem): content(&mem) {}

Symbol::Symbol(Induction& mem): content(&mem) {}

class ClassVisitor
{
	public:
	ClassVisitor(SymbolTable* table) : table(table)
	{
		assert(this->table != nullptr);
	}

	void operator()(Function& function)
	{
		table->addSymbol(function);

		for (auto& member : function.getMembers())
			table->addSymbol(member);
	}

	void operator()(Class& model)
	{
		table->addSymbol(model);

		for (auto& member : model.getMembers())
			table->addSymbol(member);

		for (auto& cls : model.getInnerClasses())
			cls->visit(*this);
	}

	private:
	SymbolTable* table;
};


SymbolTable::SymbolTable() {}

SymbolTable::SymbolTable(const SymbolTable* parent): parentTable(parent) {}

SymbolTable::SymbolTable(Function& function, const SymbolTable* parent)
		: parentTable(parent)
{
	addSymbol(function);

	for (auto& member : function.getMembers())
		addSymbol(member);
}

SymbolTable::SymbolTable(Class& model, const SymbolTable* parent)
		: parentTable(parent)
{
	addSymbol(model);

	for (auto& member : model.getMembers())
		addSymbol(member);

	for (auto& cls : model.getInnerClasses())
		cls->visit(ClassVisitor(this));
}

bool SymbolTable::hasSymbol(llvm::StringRef name) const
{
	if (table.find(name) != table.end())
		return true;

	if (parentTable == nullptr)
		return false;

	return parentTable->hasSymbol(name);
}
