set(SOURCES
		src/Main.cpp)

marco_add_tool(modc ${SOURCES})

target_link_libraries(modc
		PRIVATE marco::model
		PRIVATE marco::lowerer)

add_subdirectory(test)
