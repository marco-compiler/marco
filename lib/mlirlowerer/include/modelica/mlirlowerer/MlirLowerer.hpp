#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/utils/SourceRange.hpp>
#include <mlir/InitAllDialects.h>

namespace modelica
{
	class MlirLowerer
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		explicit MlirLowerer(mlir::MLIRContext &context);

		mlir::FuncOp lower(const modelica::ClassContainer& cls);
		mlir::FuncOp lower(const modelica::Class& cls);
		mlir::FuncOp lower(const modelica::Function& function);
		mlir::Type lower(const modelica::Type& type);
		mlir::Type lower(const modelica::BuiltInType& type);
		mlir::Type lower(const modelica::UserDefinedType& type);

		mlir::LogicalResult lower(const modelica::Algorithm& algorithm);

		Container<std::pair<llvm::StringRef, mlir::Value>> lower(const modelica::Statement& statement);
		Container<std::pair<llvm::StringRef, mlir::Value>> lower(const modelica::AssignmentStatement& statement);
		Container<std::pair<llvm::StringRef, mlir::Value>> lower(const modelica::IfStatement& statement);
		Container<std::pair<llvm::StringRef, mlir::Value>> lower(const modelica::ForStatement& statement);
		Container<std::pair<llvm::StringRef, mlir::Value>> lower(const modelica::WhileStatement& statement);
		Container<std::pair<llvm::StringRef, mlir::Value>> lower(const modelica::WhenStatement& statement);
		Container<std::pair<llvm::StringRef, mlir::Value>> lower(const modelica::BreakStatement& statement);
		Container<std::pair<llvm::StringRef, mlir::Value>> lower(const modelica::ReturnStatement& statement);

		template<typename T>
		Container<mlir::Value> lower(const modelica::Expression& expression);

		public:
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

		mlir::Type constantToType(const Constant& constant)
		{
			if (constant.isA<BuiltInType::Boolean>())
				return builder.getI1Type();

			if (constant.isA<BuiltInType::Integer>())
				return builder.getI32Type();

			if (constant.isA<BuiltInType::Float>())
				return builder.getF32Type();

			assert(false && "Unreachable");
		}

		template<typename T>
		constexpr mlir::Attribute getAttribute(const T& value)
		{
			constexpr BuiltInType type = typeToFrontendType<T>();

			if constexpr (type == BuiltInType::Boolean)
				return builder.getBoolAttr(value);
			else if constexpr (type == BuiltInType::Integer)
				return builder.getI32IntegerAttr(value);
			else if constexpr (type == BuiltInType::Float)
				return builder.getF32FloatAttr(value);

			assert(false && "Unreachable");
		}
	};

	template<>
	MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Expression>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Operation>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Constant>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::ReferenceAccess>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Call>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Tuple>(const modelica::Expression& expression);
}
