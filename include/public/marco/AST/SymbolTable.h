#ifndef MARCO_AST_SYMBOLTABLE_H
#define MARCO_AST_SYMBOLTABLE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "marco/AST/AST.h"
#include "marco/AST/Symbol.h"
#include <variant>

namespace marco::ast
{
	class SymbolTable
	{
		public:
		// Notice we provide a constructor that takes a pointer, because otherwise
		// we would be shadowing the copy constructor.
		SymbolTable();
		explicit SymbolTable(const SymbolTable* parent);
		explicit SymbolTable(Class& cls, const SymbolTable* parent = nullptr);

		[[nodiscard]] const Symbol& operator[](llvm::StringRef name) const
		{
			assert(hasSymbol(name));

			if (auto iter = table.find(name); iter != table.end())
				return iter->second;

			return (*parentTable)[name];
		}

		[[nodiscard]] bool hasSymbol(llvm::StringRef name) const;

		template<typename T>
		void addSymbol(T& s)
		{
			table.try_emplace(s.getName(), Symbol(s));
		}

		private:
		const SymbolTable* parentTable{ nullptr };
		llvm::StringMap<Symbol> table;
	};
}

#endif // MARCO_AST_SYMBOLTABLE_H
