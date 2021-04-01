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
	class ReturnStatement;
	class Statement;
	class WhenStatement;
	class WhileStatement;

	class ReturnRemover: public Pass
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
	};

	template<>
	bool ReturnRemover::run<Statement>(Statement& statement);

	template<>
	bool ReturnRemover::run<AssignmentStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<IfStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<ForStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<WhileStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<WhenStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<BreakStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<ReturnStatement>(Statement& statement);

	std::unique_ptr<Pass> createReturnRemovingPass();
}
