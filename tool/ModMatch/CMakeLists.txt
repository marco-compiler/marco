set(SOURCES
		src/Main.cpp)

marco_add_tool(modmatch ${SOURCES})

target_link_libraries(modmatch
		PRIVATE marco::model
		PRIVATE marco::lowerer)

add_subdirectory(test)
