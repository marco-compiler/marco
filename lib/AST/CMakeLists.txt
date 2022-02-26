set(SOURCES
		Node/Algorithm.cpp
		Node/Annotation.cpp
		Node/Array.cpp
		Node/ASTNode.cpp
		Node/Call.cpp
		Node/Class.cpp
		Node/Model.cpp
		Node/Constant.cpp
		Node/Equation.cpp
		Node/Expression.cpp
		Node/ForEquation.cpp
		Node/Function.cpp
		Node/Induction.cpp
		Node/Member.cpp
		Node/Modification.cpp
		Node/Operation.cpp
		Node/Package.cpp
		Node/Record.cpp
		Node/ReferenceAccess.cpp
		Node/Statement.cpp
		Node/Tuple.cpp
		Node/Type.cpp
		Node/TypePrefix.cpp
		Passes/ConstantFoldingPass.cpp
		Passes/PassManager.cpp
		Passes/TypeCheckingPass.cpp
		Errors.cpp
		LexerStateMachine.cpp
		Parser.cpp
		Symbol.cpp
		SymbolTable.cpp)

marco_add_library(ast SHARED
		${SOURCES}

		LINK_LIBS
		PRIVATE marco::utils
		PRIVATE boost

		LINK_COMPONENTS
		Core
		Support)
