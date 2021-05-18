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

		template<typename T>
		llvm::Error run(Class& cls);

		template<typename T>
		bool run(Statement& statement);

		llvm::Error run(Algorithm& algorithm);
	};

	template<>
	llvm::Error ReturnRemover::run<DerFunction>(Class& cls);

	template<>
	llvm::Error ReturnRemover::run<StandardFunction>(Class& cls);

	template<>
	llvm::Error ReturnRemover::run<Model>(Class& cls);

	template<>
	llvm::Error ReturnRemover::run<Package>(Class& cls);

	template<>
	llvm::Error ReturnRemover::run<Record>(Class& cls);

	template<>
	bool ReturnRemover::run<AssignmentStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<BreakStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<ForStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<IfStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<ReturnStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<WhenStatement>(Statement& statement);

	template<>
	bool ReturnRemover::run<WhileStatement>(Statement& statement);

	std::unique_ptr<Pass> createReturnRemovingPass();
}
