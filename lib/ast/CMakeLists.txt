set(SOURCES
		src/nodes/Algorithm.cpp
		src/nodes/Annotation.cpp
		src/nodes/Array.cpp
		src/nodes/ASTNode.cpp
		src/nodes/Call.cpp
		src/nodes/Class.cpp
		src/nodes/Model.cpp
		src/nodes/Constant.cpp
		src/nodes/Equation.cpp
		src/nodes/Expression.cpp
		src/nodes/ForEquation.cpp
		src/nodes/Function.cpp
		src/nodes/Induction.cpp
		src/nodes/Member.cpp
		src/nodes/Modification.cpp
		src/nodes/Operation.cpp
		src/nodes/Package.cpp
		src/nodes/Record.cpp
		src/nodes/ReferenceAccess.cpp
		src/nodes/Statement.cpp
		src/nodes/Tuple.cpp
		src/nodes/Type.cpp
		src/nodes/TypePrefix.cpp
		src/passes/ConstantFoldingPass.cpp
		src/passes/PassManager.cpp
		src/passes/TypeCheckingPass.cpp
		src/Errors.cpp
		src/LexerStateMachine.cpp
		src/Parser.cpp
		src/Symbol.cpp
		src/SymbolTable.cpp)

marco_add_library(ast SHARED ${SOURCES})

marco_link_llvm_libs(ast USE_SHARED core support)

target_link_libraries(ast
		PUBLIC marco::utils
		PUBLIC boost)

add_subdirectory(test)
