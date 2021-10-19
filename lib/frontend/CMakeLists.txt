set(SOURCES
		src/ast/Algorithm.cpp
		src/ast/Annotation.cpp
		src/ast/Array.cpp
		src/ast/ASTNode.cpp
		src/ast/Call.cpp
		src/ast/Class.cpp
		src/ast/Model.cpp
		src/ast/Constant.cpp
		src/ast/Equation.cpp
		src/ast/Expression.cpp
		src/ast/ForEquation.cpp
		src/ast/Function.cpp
		src/ast/Induction.cpp
		src/ast/Member.cpp
		src/ast/Modification.cpp
		src/ast/Operation.cpp
		src/ast/Package.cpp
		src/ast/Record.cpp
		src/ast/ReferenceAccess.cpp
		src/ast/Statement.cpp
		src/ast/Tuple.cpp
		src/ast/Type.cpp
		src/ast/TypePrefix.cpp
		src/passes/ConstantFoldingPass.cpp
		src/passes/PassManager.cpp
		src/passes/TypeCheckingPass.cpp
		src/Errors.cpp
		src/LexerStateMachine.cpp
		src/Parser.cpp
		src/Symbol.cpp
		src/SymbolTable.cpp)

marco_add_library(frontend SHARED ${SOURCES})

marco_link_llvm_libs(frontend USE_SHARED core support)

target_link_libraries(frontend
		PUBLIC marco::utils
		PUBLIC boost)

add_subdirectory(test)
