#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <modelica/frontend/Class.hpp>
#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/frontend/ForEquation.hpp>
#include <modelica/frontend/Function.hpp>
#include <modelica/frontend/Member.hpp>
#include <variant>

namespace modelica
{
	class Symbol
	{
		public:
		explicit Symbol(Function& function);
		explicit Symbol(Class& model);
		explicit Symbol(Member& mem);
		explicit Symbol(Induction& mem);

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
		std::variant<Function*, Class*, Member*, Induction*> content;
	};

	class SymbolTable
	{
		public:
		// Notice we provide a constructor that takes a pointer, because otherwise
		// we would be shadowing the copy constructor.
		SymbolTable();
		explicit SymbolTable(const SymbolTable* parent);
		explicit SymbolTable(Function& cls, const SymbolTable* parent = nullptr);
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
}	 // namespace modelica
