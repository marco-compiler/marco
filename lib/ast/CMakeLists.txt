set(SOURCES
		nodes/Algorithm.cpp
		nodes/Annotation.cpp
		nodes/Array.cpp
		nodes/ASTNode.cpp
		nodes/Call.cpp
		nodes/Class.cpp
		nodes/Model.cpp
		nodes/Constant.cpp
		nodes/Equation.cpp
		nodes/Expression.cpp
		nodes/ForEquation.cpp
		nodes/Function.cpp
		nodes/Induction.cpp
		nodes/Member.cpp
		nodes/Modification.cpp
		nodes/Operation.cpp
		nodes/Package.cpp
		nodes/Record.cpp
		nodes/ReferenceAccess.cpp
		nodes/Statement.cpp
		nodes/Tuple.cpp
		nodes/Type.cpp
		nodes/TypePrefix.cpp
		passes/ConstantFoldingPass.cpp
		passes/PassManager.cpp
		passes/TypeCheckingPass.cpp
		Errors.cpp
		LexerStateMachine.cpp
		Parser.cpp
		Symbol.cpp
		SymbolTable.cpp)

marco_add_library(ast STATIC
		${SOURCES}

		LINK_LIBS
		PRIVATE marco::utils
		PRIVATE boost

		LINK_COMPONENTS
		Core
		Support)
