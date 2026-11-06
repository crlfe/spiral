#pragma once

#ifdef TRACE
#include <aie_api/aie.hpp>
#define TRACE_ENTER() event0()
#define TRACE_LEAVE() event1()
#else
#define TRACE_ENTER() do {} while (0)
#define TRACE_LEAVE() do {} while (0)
#endif
