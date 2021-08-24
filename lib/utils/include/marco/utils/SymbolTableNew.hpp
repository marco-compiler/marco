#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <marco/frontend/Class.hpp>
#include <marco/frontend/ClassContainer.hpp>
#include <marco/frontend/ForEquation.hpp>
#include <marco/frontend/Function.hpp>
#include <marco/frontend/Member.hpp>
#include <variant>

namespace marco
{
	template<class Symbol>
	class SymbolTable
	{
		public:
		SymbolTable() = default;

		explicit SymbolTable(const SymbolTable* parent) : parent(parent)
		{
		}

		[[nodiscard]] const Symbol& operator[](llvm::StringRef name) const
		{
			assert(hasSymbol(name));

			if (auto iter = table.find(name); iter != table.end())
				return iter->second;

			return (*parent)[name];
		}

		[[nodiscard]] bool hasSymbol(llvm::StringRef name) const
		{
			if (table.find(name) != table.end())
				return true;

			if (parent == nullptr)
				return false;

			return parent->hasSymbol(name);
		}

		template<typename T>
		void addSymbol(T& s)
		{
			table.try_emplace(s.getName(), Symbol(s));
		}

		private:
		const SymbolTable* parent{ nullptr };
		llvm::StringMap<Symbol> table;
	};
}	 // namespace marco
