#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <modelica/frontend/Pass.h>
#include <modelica/frontend/SymbolTable.hpp>

namespace modelica::frontend
{
	class Call;
	class Class;
	class ClassContainer;
	class Equation;
	class Expression;
	class ForEquation;
	class Function;
	class Member;

	class ConstantFolder : public Pass
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
		using SymbolTableScope = llvm::ScopedHashTableScope<llvm::StringRef, Symbol>;

		llvm::Error run(ClassContainer& cls);
		llvm::Error run(Class& cls);
		llvm::Error run(Function& function);
		llvm::Error run(Package& package);
		llvm::Error run(Record& record);
		llvm::Error run(Member& member);
		llvm::Error run(Equation& equation);
		llvm::Error run(ForEquation& forEquation);
		llvm::Error run(Expression& expression);
		llvm::Error run(Call& call);

		llvm::Error foldReference(Expression& exp);
		llvm::Expected<Expression> foldExpression(Expression& exp);
		llvm::Expected<Expression> foldOpSubscrition(Expression& exp);

		SymbolTable& getSymbolTable()
		{
			return symbolTable;
		}

		private:
		llvm::ScopedHashTable<llvm::StringRef, Symbol> symbolTable;
	};

	std::unique_ptr<Pass> createConstantFolderPass();
}
