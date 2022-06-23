set(SOURCES
	LogMessage.cpp)

marco_add_library(utils
	${SOURCES}

	LINK_COMPONENTS
	Core
	Support)
