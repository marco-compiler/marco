#include <modelica/frontend/SymbolTable.hpp>

using namespace llvm;
using namespace modelica::frontend;
using namespace std;

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
			table->addSymbol(*member);
	}

	void operator()(Class& model)
	{
		table->addSymbol(model);

		for (auto& member : model.getMembers())
			table->addSymbol(*member);

		for (auto& cls : model.getInnerClasses())
			cls->visit(*this);
	}

	void operator()(Package& package)
	{
		table->addSymbol(package);

		for (auto& cls : package)
			cls.visit(*this);
	}

	void operator()(Record& record)
	{
		table->addSymbol(record);

		for (auto& member : record)
			table->addSymbol(member);
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
		addSymbol(*member);
}

SymbolTable::SymbolTable(Class& model, const SymbolTable* parent)
		: parentTable(parent)
{
	addSymbol(model);

	for (auto& member : model.getMembers())
		addSymbol(*member);

	for (auto& cls : model.getInnerClasses())
		cls->visit(ClassVisitor(this));
}

SymbolTable::SymbolTable(Package& package, const SymbolTable* parent)
		: parentTable(parent)
{
	addSymbol(package);

	for (auto& cls : package)
		cls.visit(ClassVisitor(this));
}


SymbolTable::SymbolTable(Record& record, const SymbolTable* parent)
		: parentTable(parent)
{
	addSymbol(record);

	for (auto& member : record)
		addSymbol(member);
}

bool SymbolTable::hasSymbol(llvm::StringRef name) const
{
	if (table.find(name) != table.end())
		return true;

	if (parentTable == nullptr)
		return false;

	return parentTable->hasSymbol(name);
}
