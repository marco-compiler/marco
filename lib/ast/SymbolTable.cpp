#include <marco/ast/SymbolTable.h>

using namespace llvm;
using namespace marco::ast;
using namespace std;

class ClassVisitor
{
	public:
	ClassVisitor(SymbolTable* table) : table(table)
	{
		assert(this->table != nullptr);
	}

	void operator()(PartialDerFunction& function)
	{
	}

	void operator()(StandardFunction& function)
	{
		for (auto& member : function.getMembers())
			table->addSymbol(*member);
	}

	void operator()(Model& model)
	{
		for (auto& member : model.getMembers())
			table->addSymbol(*member);

		for (auto& cls : model.getInnerClasses())
		{
			table->addSymbol(*cls);
			cls->visit(*this);
		}
	}

	void operator()(Package& package)
	{
		for (auto& cls : package.getInnerClasses())
		{
			table->addSymbol(*cls);
			cls->visit(*this);
		}
	}

	void operator()(Record& record)
	{
		for (auto& member : record)
			table->addSymbol(*member);
	}

	private:
	SymbolTable* table;
};

SymbolTable::SymbolTable() {}

SymbolTable::SymbolTable(const SymbolTable* parent): parentTable(parent) {}

SymbolTable::SymbolTable(Class& cls, const SymbolTable* parent)
		: parentTable(parent)
{
	addSymbol(cls);
	cls.visit(ClassVisitor(this));
}

bool SymbolTable::hasSymbol(llvm::StringRef name) const
{
	if (table.find(name) != table.end())
		return true;

	if (parentTable == nullptr)
		return false;

	return parentTable->hasSymbol(name);
}
