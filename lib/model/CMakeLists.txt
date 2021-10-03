set(SOURCES
		src/ModExp.cpp
		src/ModErrors.cpp
		src/Assigment.cpp
		src/ModType.cpp
		src/ModCall.cpp
		src/ModLexerStateMachine.cpp
		src/ModParser.cpp
		src/Model.cpp
		src/ModEquation.cpp
		src/ModVariable.cpp
		src/VectorAccess.cpp
		src/ModConst.cpp
		src/ModMatchers.cpp)

marco_add_library(model SHARED ${SOURCES})

llvm_config(model USE_SHARED core support)

target_link_libraries(model
		PUBLIC marco::utils)

add_subdirectory(test)
