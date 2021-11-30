#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <marco/ast/Pass.h>
#include <marco/ast/SymbolTable.h>

namespace marco::ast
{
	class Algorithm;
	class AssignmentStatement;
	class BreakStatement;
	class Call;
	class Class;
	class ClassContainer;
	class Constant;
	class PartialDerFunction;
	class Equation;
	class Expression;
	class ForEquation;
	class ForStatement;
	class IfStatement;
	class Member;
	class Operation;
	class ReferenceAccess;
	class ReturnStatement;
	class StandardFunction;
	class Statement;
	class Tuple;
	class Type;
	class WhenStatement;
	class WhileStatement;

	class ConstantFolder : public Pass
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
		using SymbolTableScope = llvm::ScopedHashTableScope<llvm::StringRef, Symbol>;

		llvm::Error run(llvm::ArrayRef<std::unique_ptr<Class>> classes) final;

		llvm::Error run(Algorithm& algorithm);

		template<typename T>
		[[nodiscard]] llvm::Error run(Class& cls);

		llvm::Error run(Equation& equation);

		template<typename T>
		[[nodiscard]] llvm::Error run(Expression& expression);

		llvm::Error run(ForEquation& forEquation);
		llvm::Error run(Induction& induction);
		llvm::Error run(Member& member);

		template<typename T>
		[[nodiscard]] llvm::Error run(Statement& statement);

		[[nodiscard]] llvm::Error foldAddOp(Expression& expression);
		[[nodiscard]] llvm::Error foldDifferentOp(Expression& expression);
		[[nodiscard]] llvm::Error foldDivOp(Expression& expression);
		[[nodiscard]] llvm::Error foldEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error foldGreaterOp(Expression& expression);
		[[nodiscard]] llvm::Error foldGreaterEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error foldIfElseOp(Expression& expression);
		[[nodiscard]] llvm::Error foldLogicalAndOp(Expression& expression);
		[[nodiscard]] llvm::Error foldLogicalOrOp(Expression& expression);
		[[nodiscard]] llvm::Error foldLessOp(Expression& expression);
		[[nodiscard]] llvm::Error foldLessEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error foldMemberLookupOp(Expression& expression);
		[[nodiscard]] llvm::Error foldMulOp(Expression& expression);
		[[nodiscard]] llvm::Error foldNegateOp(Expression& expression);
		[[nodiscard]] llvm::Error foldPowerOfOp(Expression& expression);
		[[nodiscard]] llvm::Error foldSubOp(Expression& expression);
		[[nodiscard]] llvm::Error foldSubscriptionOp(Expression& expression);

		SymbolTable& getSymbolTable()
		{
			return symbolTable;
		}

		private:
		llvm::ScopedHashTable<llvm::StringRef, Symbol> symbolTable;
	};

	template<>
	llvm::Error ConstantFolder::run<Class>(Class& cls);

	template<>
	llvm::Error ConstantFolder::run<PartialDerFunction>(Class& cls);

	template<>
	llvm::Error ConstantFolder::run<StandardFunction>(Class& cls);

	template<>
	llvm::Error ConstantFolder::run<Model>(Class& cls);

	template<>
	llvm::Error ConstantFolder::run<Package>(Class& cls);

	template<>
	llvm::Error ConstantFolder::run<Record>(Class& cls);

	template<>
	llvm::Error ConstantFolder::run<Expression>(Expression& expression);

	template<>
	llvm::Error ConstantFolder::run<Array>(Expression& expression);

	template<>
	llvm::Error ConstantFolder::run<Call>(Expression& expression);

	template<>
	llvm::Error ConstantFolder::run<Constant>(Expression& expression);

	template<>
	llvm::Error ConstantFolder::run<Operation>(Expression& expression);

	template<>
	llvm::Error ConstantFolder::run<ReferenceAccess>(Expression& expression);

	template<>
	llvm::Error ConstantFolder::run<Tuple>(Expression& expression);

	template<>
	llvm::Error ConstantFolder::run<RecordInstance>(Expression& expression);

	template<>
	llvm::Error ConstantFolder::run<AssignmentStatement>(Statement& statement);

	template<>
	llvm::Error ConstantFolder::run<BreakStatement>(Statement& statement);

	template<>
	llvm::Error ConstantFolder::run<ForStatement>(Statement& statement);

	template<>
	llvm::Error ConstantFolder::run<IfStatement>(Statement& statement);

	template<>
	llvm::Error ConstantFolder::run<ReturnStatement>(Statement& statement);

	template<>
	llvm::Error ConstantFolder::run<WhenStatement>(Statement& statement);

	template<>
	llvm::Error ConstantFolder::run<WhileStatement>(Statement& statement);

	std::unique_ptr<Pass> createConstantFolderPass();
}
