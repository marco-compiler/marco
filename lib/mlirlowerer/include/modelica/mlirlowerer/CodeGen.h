#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/SymbolTable.hpp>
#include <modelica/matching/Matching.hpp>
#include <modelica/matching/SccCollapsing.hpp>
#include <modelica/matching/Schedule.hpp>
#include <modelica/model/ModType.hpp>
#include <modelica/model/Model.hpp>
#include <modelica/omcToModel/OmcToModelPass.hpp>
#include <modelica/passes/ConstantFold.hpp>
#include <modelica/passes/SolveModel.hpp>
#include <modelica/utils/SourcePosition.h>

#include "ModelicaBuilder.h"
#include "Passes.h"

namespace modelica::codegen
{
	struct ModelicaOptions {

		bool x64 = true;
		double startTime = 0;
		double endTime = 10;
		double timeStep = 0.1;

		[[nodiscard]] unsigned int getBitWidth() const
		{
			if (x64)
				return 64;

			return 32;
		}

		/**
		 * Get a statically allocated copy of the default options.
		 *
		 * @return default options
		 */
		static const ModelicaOptions& getDefaultOptions() {
			static ModelicaOptions options;
			return options;
		}
	};

	struct ModelicaLoweringOptions
	{
		SolveModelOptions solveModelOptions = SolveModelOptions::getDefaultOptions();
		bool inlining = true;
		bool resultBuffersToArgs = true;
		bool cse = true;
		bool openmp = false;
		ModelicaConversionOptions conversionOptions = ModelicaConversionOptions::getDefaultOptions();
		ModelicaToLLVMConversionOptions llvmOptions = ModelicaToLLVMConversionOptions::getDefaultOptions();
		bool debug = true;

		static const ModelicaLoweringOptions& getDefaultOptions() {
			static ModelicaLoweringOptions options;
			return options;
		}
	};

	class Reference
	{
		public:
		Reference();

		[[nodiscard]] mlir::Value operator*();
		[[nodiscard]] mlir::Value getReference() const;
		[[nodiscard]] bool isInitialized() const;

		void set(mlir::Value value);

		[[nodiscard]] static Reference ssa(mlir::OpBuilder* builder, mlir::Value value);
		[[nodiscard]] static Reference memory(mlir::OpBuilder* builder, mlir::Value value, bool initialized);
		[[nodiscard]] static Reference member(mlir::OpBuilder* builder, mlir::Value value, bool initialized);

		private:
		Reference(mlir::OpBuilder* builder,
							mlir::Value value,
							bool initialized,
							std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader,
							std::function<void(mlir::OpBuilder* builder, Reference& destination, mlir::Value)> writer);

		mlir::OpBuilder* builder;
		mlir::Value value;
		bool initialized;
		std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader;
		std::function<void(mlir::OpBuilder*, Reference&, mlir::Value)> writer;
	};

	class MLIRLowerer
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		explicit MLIRLowerer(mlir::MLIRContext& context, ModelicaOptions options = ModelicaOptions::getDefaultOptions());

		mlir::LogicalResult convertToLLVMDialect(mlir::ModuleOp& module, ModelicaLoweringOptions options = ModelicaLoweringOptions::getDefaultOptions());

		llvm::Optional<mlir::ModuleOp> run(llvm::ArrayRef<std::unique_ptr<frontend::Class>> classes);

		private:
		mlir::Operation* lower(const frontend::Class& cls);
		mlir::Operation* lower(const frontend::DerFunction& function);
		mlir::Operation* lower(const frontend::StandardFunction& function);
		mlir::Operation* lower(const frontend::Model& model);
		mlir::Operation* lower(const frontend::Package& package);
		mlir::Operation* lower(const frontend::Record& record);

		mlir::Type lower(const frontend::Type& type, BufferAllocationScope allocationScope);
		mlir::Type lower(const frontend::BuiltInType& type, BufferAllocationScope allocationScope);
		mlir::Type lower(const frontend::PackedType& type, BufferAllocationScope allocationScope);
		mlir::Type lower(const frontend::UserDefinedType& type, BufferAllocationScope allocationScope);

		template<typename Context>
		void lower(const frontend::Member& member);

		void lower(const frontend::Equation& equation);
		void lower(const frontend::ForEquation& forEquation);

		void lower(const frontend::Algorithm& algorithm);
		void lower(const frontend::Statement& statement);
		void lower(const frontend::AssignmentStatement& statement);
		void lower(const frontend::IfStatement& statement);
		void lower(const frontend::ForStatement& statement);
		void lower(const frontend::WhileStatement& statement);
		void lower(const frontend::WhenStatement& statement);
		void lower(const frontend::BreakStatement& statement);
		void lower(const frontend::ReturnStatement& statement);

		template<typename T>
		Container<Reference> lower(const frontend::Expression& expression);

		/**
		 * The builder is a helper class to create IR inside a function. The
		 * builder is stateful, in particular it keeps an "insertion point":
		 * this is where the next operations will be introduced.
		 */
		ModelicaBuilder builder;

		/**
		 * The symbol table maps a variable name to a value in the current scope.
		 * Entering a function creates a new scope, and the function arguments
		 * are added to the mapping. When the processing of a function is
		 * terminated, the scope is destroyed and the mappings created in this
		 * scope are dropped.
		 */
		llvm::ScopedHashTable<llvm::StringRef, Reference> symbolTable;

		/**
		 * The stack represent the list of the nested scope names in which the
		 * lowerer currently is.
		 */
		std::deque<llvm::StringRef> scopes;

		 /**
		  * Apply a binary operation to a list of values.
		  *
		  * @param args      arguments
		  * @param callback  callback that should process the current args and return a result
		  * @return folded value
		  */
		 mlir::Value foldBinaryOperation(llvm::ArrayRef<mlir::Value> args, std::function<mlir::Value(mlir::Value, mlir::Value)> callback);

		/**
		 * Lower the arguments of an operation.
		 *
		 * @param operation operation whose arguments have to be lowered
		 * @return lowered args
 		 */
		Container<mlir::Value> lowerOperationArgs(const frontend::Operation& operation);

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourcePosition location);

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourceRange location);

		ModelicaOptions options;
	};

	template<>
	void MLIRLowerer::lower<frontend::Model>(
			const frontend::Member& member);

	template<>
	void MLIRLowerer::lower<frontend::Function>(
			const frontend::Member& member);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Expression>(
			const frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Operation>(
			const frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Constant>(
			const frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::ReferenceAccess>(
			const frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Call>(
			const frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Tuple>(
			const frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Array>(
			const frontend::Expression& expression);
}
