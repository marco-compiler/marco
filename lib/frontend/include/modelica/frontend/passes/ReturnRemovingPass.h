#pragma once

#include <memory>
#include <modelica/frontend/Pass.h>

namespace modelica::frontend
{
	class Algorithm;
	class AssignmentStatement;
	class BreakStatement;
	class Class;
	class Function;
	class IfStatement;
	class ForStatement;
	class Model;
	class Package;
	class Record;
	class ReturnStatement;
	class Statement;
	class WhenStatement;
	class WhileStatement;

	class ReturnRemover: public Pass
	{
		public:
		llvm::Error run(Class& cls) final;
		llvm::Error run(Function& function);
		llvm::Error run(DerFunction& function);
		llvm::Error run(StandardFunction& function);
		llvm::Error run(Model& cls);
		llvm::Error run(Package& package);
		llvm::Error run(Record& record);
		llvm::Error run(Algorithm& algorithm);
		bool run(Statement& statement);
		bool run(AssignmentStatement& statement);
		bool run(BreakStatement& statement);
		bool run(ForStatement& statement);
		bool run(IfStatement& statement);
		bool run(ReturnStatement& statement);
		bool run(WhenStatement& statement);
		bool run(WhileStatement& statement);

		private:
	};

	std::unique_ptr<Pass> createReturnRemovingPass();
}
