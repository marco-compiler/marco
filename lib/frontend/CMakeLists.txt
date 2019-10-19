ADD_LIBRARY(frontend SHARED src/Dumper.cpp src/LexerStateMachine.cpp src/Parser.cpp src/ParserEquation.cpp src/ParserErrors.cpp src/Expr.cpp src/Equation.cpp src/ParserStatement.cpp src/ParserClasses.cpp src/Statement.cpp)
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

TARGET_LINK_LIBRARIES(frontend PUBLIC modelica::model)

modelicaInstall(frontend)

ADD_SUBDIRECTORY(test)
