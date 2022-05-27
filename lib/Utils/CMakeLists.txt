set(SOURCES
	Lexer.cpp
	LogMessage.cpp)

marco_add_library(utils SHARED
	${SOURCES}

	LINK_COMPONENTS
	Core
	Support)
