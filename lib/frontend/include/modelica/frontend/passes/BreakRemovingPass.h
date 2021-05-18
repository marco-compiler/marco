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

	class BreakRemover: public Pass
	{
		public:
		llvm::Error run(Class& cls) final;

		template<typename T>
		llvm::Error run(Class& cls);

		template<typename T>
		bool run(Statement& statement);

		llvm::Error run(Algorithm& algorithm);

		private:
		int nestLevel = 0;
	};

	template<>
	llvm::Error BreakRemover::run<DerFunction>(Class& cls);

	template<>
	llvm::Error BreakRemover::run<StandardFunction>(Class& cls);

	template<>
	llvm::Error BreakRemover::run<Model>(Class& cls);

	template<>
	llvm::Error BreakRemover::run<Package>(Class& cls);

	template<>
	llvm::Error BreakRemover::run<Record>(Class& cls);

	template<>
	bool BreakRemover::run<AssignmentStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<BreakStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<ForStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<IfStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<ReturnStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<WhenStatement>(Statement& statement);

	template<>
	bool BreakRemover::run<WhileStatement>(Statement& statement);

	std::unique_ptr<Pass> createBreakRemovingPass();
}
