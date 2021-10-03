set(SOURCES
		src/Main.cpp)

marco_add_tool(ASTDumper ${SOURCES})

target_link_libraries(ASTDumper
		PRIVATE marco::frontend)

add_subdirectory(test)
