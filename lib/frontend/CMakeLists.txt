modelicaAddLibrary(
        frontend SHARED
		src/ast/Algorithm.cpp
		src/ast/Annotation.cpp
		src/ast/Call.cpp
		src/ast/Class.cpp
		src/ast/ClassContainer.cpp
		src/ast/Constant.cpp
		src/ast/Equation.cpp
		src/ast/Expression.cpp
		src/ast/ForEquation.cpp
		src/ast/Function.cpp
		src/ast/Induction.cpp
		src/ast/Member.cpp
		src/ast/Modification.cpp
		src/ast/Operation.cpp
		src/ast/ReferenceAccess.cpp
		src/ast/Statement.cpp
		src/ast/Tuple.cpp
		src/ast/Type.cpp
		src/ast/TypePrefix.cpp
		src/passes/BreakRemovingPass.cpp
		src/passes/ConstantFoldingPass.cpp
		src/passes/PassManager.cpp
		src/passes/ReturnRemovingPass.cpp
		src/passes/TypeCheckingPass.cpp
        src/LexerStateMachine.cpp
        src/Parser.cpp
        src/ParserErrors.cpp
		src/Symbol.cpp
		src/SymbolTable.cpp

)

target_link_libraries(frontend PUBLIC modelica::model boost)
