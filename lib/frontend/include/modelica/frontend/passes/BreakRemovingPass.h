#pragma once

#include <memory>
#include <modelica/frontend/Pass.h>

namespace modelica::frontend
{
	class Algorithm;
	class AssignmentStatement;
	class BreakStatement;
	class Class;
	class ClassContainer;
	class Function;
	class IfStatement;
	class ForStatement;
	class Package;
	class Record;
	class ReturnStatement;
	class Statement;
	class WhenStatement;
	class WhileStatement;

	class BreakRemover: public Pass
	{
		public:
		llvm::Error run(ClassContainer& cls) final;
		llvm::Error run(Class& cls);
		llvm::Error run(Function& function);
		llvm::Error run(Package& package);
		llvm::Error run(Record& record);
		llvm::Error run(Algorithm& algorithm);

		private:
		template<typename T>
		bool run(Statement& statement);

		int nestLevel = 0;
	};

	template<>
	bool BreakRemover::run<Statement>(Statement& statement);

	template<>
	bool BreakRemover::run<AssignmentStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<IfStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<ForStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<WhileStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<WhenStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<BreakStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<ReturnStatement>(Statement& statement);

	std::unique_ptr<Pass> createBreakRemovingPass();
}
