#pragma once

#include <variant>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "modelica/frontend/Class.hpp"
#include "modelica/frontend/ForEquation.hpp"
#include "modelica/frontend/Member.hpp"
namespace modelica
{
	class Symbol
	{
		public:
		explicit Symbol(Class& clas): content(&clas) {}
		explicit Symbol(Member& mem): content(&mem) {}
		explicit Symbol(Induction& mem): content(&mem) {}

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T*>(content);
		}
		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(isA<T>());
			return *std::get<T*>(content);
		}

		private:
		std::variant<Class*, Member*, Induction*> content;
	};

	class SymbolTable
	{
		public:
		// notice we provide a constructor that takes a pointer because otherwise we
		// would be shadowing the copy constructor
		explicit SymbolTable(const SymbolTable* parent): parentTable(parent) {}
		SymbolTable() = default;
		explicit SymbolTable(Class& cls, const SymbolTable* parent = nullptr)
				: parentTable(parent)
		{
			addSymbol(cls);
			for (auto& member : cls.getMembers())
				addSymbol(member);
		}

		template<typename T>
		void addSymbol(T& s)
		{
			table.try_emplace(s.getName(), Symbol(s));
		}

		[[nodiscard]] const Symbol& operator[](llvm::StringRef name) const
		{
			assert(hasSymbol(name));
			if (auto iter = table.find(name); iter != table.end())
				return iter->second;

			return (*parentTable)[name];
		}

		[[nodiscard]] bool hasSymbol(llvm::StringRef name) const
		{
			if (table.find(name) != table.end())
				return true;
			if (parentTable == nullptr)
				return false;
			return parentTable->hasSymbol(name);
		}

		private:
		const SymbolTable* parentTable{ nullptr };
		llvm::StringMap<Symbol> table;
	};
}	 // namespace modelica
