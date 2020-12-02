#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/utils/SourceRange.hpp>

namespace modelica
{
	class MlirLowerer
	{
		public:
		MlirLowerer(mlir::MLIRContext &context);

		mlir::FuncOp lower(ClassContainer cls);
		mlir::FuncOp lower(Class cls);
		mlir::FuncOp lower(Function function);
		mlir::Type lower(Type type);
		mlir::Type lower(BuiltInType type);
		mlir::Type lower(UserDefinedType type);
		void lower(Algorithm algorithm);
		void lower(Statement statement);
		void lower(AssignmentStatement statement);
		void lower(IfStatement statement);
		void lower(ForStatement statement);
		void lower(WhileStatement statement);
		void lower(WhenStatement statement);

		private:
		/// The builder is a helper class to create IR inside a function. The
		/// builder is stateful, in particular it keeps an "insertion point":
		/// this is where the next operations will be introduced.
		mlir::OpBuilder builder;

		/// The symbol table maps a variable name to a value in the current scope.
		/// Entering a function creates a new scope, and the function arguments
		/// are added to the mapping. When the processing of a function is
		/// terminated, the scope is destroyed and the mappings created in this
		/// scope are dropped.
		llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourcePosition location);
	};
}
