ADD_LIBRARY(frontend STATIC src/Lexer.cpp src/Dumper.cpp src/LexerStateMachine.cpp src/Parser.cpp src/ParserEquation.cpp src/ParserErrors.cpp src/Expr.cpp src/Equation.cpp src/ParserStatement.cpp src/ParserClasses.cpp src/Statement.cpp)
ADD_LIBRARY(modelica::frontend ALIAS frontend)
TARGET_INCLUDE_DIRECTORIES(frontend 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
TARGET_COMPILE_FEATURES(frontend PUBLIC cxx_std_17)

llvm_map_components_to_libnames(llvm_frontend_libs core support ${LLVM_TARGETS_TO_BUILD})
TARGET_LINK_LIBRARIES(frontend PUBLIC ${llvm_frontend_libs} modelica::utils)

modelicaInstall(frontend)

ADD_SUBDIRECTORY(test)
